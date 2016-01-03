from __future__ import division
import numpy as np
from numpy import newaxis as na

from pyhsmm.util.general import rcumsum
from pyhsmm.util.profiling import line_profiled

from pyhsmm.internals.hsmm_states import HSMMStatesPython, HSMMStatesPossibleChangepoints, \
        hsmm_messages_forwards_log, hsmm_messages_backwards_log

PROFILING=False

TRUNC = 3

class HSMMSubHMMStates(HSMMStatesPython):
    # NOTE: can't extend the eigen version because its sample_forwards depends
    # on aBl being iid (doesnt call the sub-methods)
    def __init__(self,model,substateseqs=None,**kwargs):
        self.model = model
        if substateseqs is not None:
            raise NotImplementedError
        super(HSMMSubHMMStates,self).__init__(model,**kwargs)
        self.data = self.data.astype('float32',copy=False) if self.data is not None else None

    def generate_states(self):
        self._generate_superstates()
        self._generate_substates()

    def _generate_superstates(self):
        super(HSMMSubHMMStates,self).generate_states()

    def _generate_substates(self):
        self.substates_list = []
        for state, dur in zip(self.stateseq_norep,self.durations_censored):
            self.model.HMMs[state].generate(dur)
            self.substates_list.append(self.model.HMMs[state].states_list[-1])

    def generate_obs(self):
        obs = []
        for subseq in self.substates_list:
            obs.append(subseq.data)
        obs = np.concatenate(obs)
        assert len(obs) == self.T
        return obs

    @property
    def aBls(self):
        if self._aBls is None:
            self._aBls = [hmm.get_aBl(self.data) for hmm in self.model.HMMs]
        return self._aBls

    @property
    def mf_aBls(self):
        if self._mf_aBls is None:
            self._mf_aBls = [hmm.get_mf_aBl(self.data) for hmm in self.model.HMMs]
        return self._mf_aBls

    def clear_caches(self):
        for hmm in self.model.HMMs:
            hmm._clear_message_caches() # NOTE: this is REALLY important!
        self._aBls = self._mf_aBls = None
        super(HSMMSubHMMStates,self).clear_caches()

    def resample(self,temp=None):
        self._remove_substates_from_subHMMs()
        super(HSMMSubHMMStates,self).resample() # resamples superstates
        self._resample_substates()

    def _resample_substates(self):
        assert not hasattr(self,'substates_list') or len(self.substates_list) == 0
        self.substates_list = []
        indices = np.concatenate(((0,),np.cumsum(self.durations_censored[:-1])))
        for state, startidx, dur in zip(self.stateseq_norep,indices,self.durations_censored):
            self.model.HMMs[state].add_data(
                    self.data[startidx:startidx+dur],initialize_from_prior=False)
            self.substates_list.append(self.model.HMMs[state].states_list[-1])

    def _remove_substates_from_subHMMs(self):
        if hasattr(self,'substates_list') and len(self.substates_list) > 0:
            for superstate, states_obj in zip(self.stateseq_norep, self.substates_list):
                self.model.HMMs[superstate].states_list.remove(states_obj)
            self.substates_list = []

    def set_stateseq(self,superstateseq,substateseqs):
        self.stateseq = superstateseq
        indices = np.concatenate(((0,),np.cumsum(self.durations_censored[:-1])))
        for state, startidx, dur, substateseq in zip(self.stateseq_norep,indices,
                self.durations_censored,substateseqs):
            self.model.HMMs[state].add_data(
                    self.data[startidx:startidx+dur],stateseq=substateseq)
            self.substates_list.append(self.model.HMMs[state].states_list[-1])

    ### NEW

    def cumulative_obs_potentials(self,t):
        return np.hstack([hmm.cumulative_obs_potentials(self.aBls[state][t:])[:,na]
            for state, hmm in enumerate(self.model.HMMs)]), np.zeros(self.num_states)
        # return np.hstack([np.logaddexp.reduce(
        #     hmm.messages_forwards(self.aBls[state][t:]),axis=1)[:,na]
        #     for state, hmm in enumerate(self.model.HMMs)])

    def reverse_cumulative_obs_potentials(self,t):
        return np.hstack([hmm.reverse_cumulative_obs_potentials(self.aBls[state][:t+1])[:,na]
            for state, hmm in enumerate(self.model.HMMs)]), np.zeros(self.num_states)
        # return np.hstack([np.logaddexp.reduce(
        #     np.log(hmm.init_state_distn.pi_0) + # could cache this
        #     hmm.messages_backwards(self.aBls[state][:t+1]),axis=1)[:,na]
        #     for state, hmm in enumerate(self.model.HMMs)])

    def mf_cumulative_obs_potentials(self,t):
        return np.hstck([hmm.mf_cumulative_obs_potentials(self.mf_aBls[state][t:])[:,na]
            for state, hmm in enumerate(self.model.HMMs)])

    def mf_reverse_cumulative_obs_potentials(self,t):
        return np.hstack([hmm.mf_reverse_cumulative_obs_potentials(self.mf_aBls[state][:t+1])[:,na]
            for state, hmm in enumerate(self.model.HMMs)])


    ### OLD

    def cumulative_likelihood_state(self,start,stop,state):
        return np.logaddexp.reduce(self.model.HMMs[state].messages_forwards(self.aBls[state][start:stop]),axis=1)

    def cumulative_likelihoods(self,start,stop):
        return np.hstack([self.cumulative_likelihood_state(start,stop,state)[:,na]
            for state in range(self.num_states)])

    def likelihood_block_state(self,start,stop,state):
        return self.model.HMMs[state].cumulative_obs_potentials(self.aBls[state][start:stop])[-1]
        # return np.logaddexp.reduce(self.model.HMMs[state].messages_forwards(self.aBls[state][start:stop])[-1])

    def likelihood_block(self,start,stop):
        return np.array([self.likelihood_block_state(start,stop,state)
            for state in range(self.num_states)])

class HSMMSubHMMStatesPossibleChangepoints(HSMMSubHMMStates,HSMMStatesPossibleChangepoints):
    # need this method as long as we don't have a general sample forwards (which
    # we probably don't want until we get the backwards normalization right...)
    def clear_caches(self):
        super(HSMMSubHMMStatesPossibleChangepoints,self).clear_caches()
        for hmm in self.model.HMMs:
            hmm._clear_message_caches()

    def sample_forwards(self,betal,betastarl):
        return HSMMStatesPossibleChangepoints.sample_forwards(self,betal,betastarl)

    def generate(self):
        raise NotImplementedError


    def cumulative_obs_potentials(self,tblock):
        t = self.segmentstarts[tblock]
        possible_durations = self.segmentlens[tblock:].cumsum()[:TRUNC]
        return np.hstack([hmm.cumulative_obs_potentials(self.aBls[state][t:],self,t)\
                [possible_durations -1][:,na]
                for state, hmm in enumerate(self.model.HMMs)]), np.zeros(self.num_states)

    def reverse_cumulative_obs_potentials(self,tblock):
        t = self.segmentstarts[tblock] + self.segmentlens[tblock]
        possible_durations = rcumsum(self.segmentlens[:tblock+1])[-TRUNC if TRUNC is not None else None:]
        return np.hstack([hmm.reverse_cumulative_obs_potentials(self.aBls[state][:t],self,t)\
                [-possible_durations][:,na]
                # [possible_durations -1][:,na]
                for state, hmm in enumerate(self.model.HMMs)])

    def mf_cumulative_obs_potentials(self,tblock):
        t = self.segmentstarts[tblock]
        possible_durations = self.segmentlens[tblock:].cumsum()[:TRUNC]
        return np.hstack([hmm.mf_cumulative_obs_potentials(self.mf_aBls[state][t:],self,t)\
                [possible_durations -1][:,na]
                for state, hmm in enumerate(self.model.HMMs)]), np.zeros(self.num_states)

    def mf_reverse_cumulative_obs_potentials(self,tblock):
        t = self.segmentstarts[tblock] + self.segmentlens[tblock]
        possible_durations = rcumsum(self.segmentlens[:tblock+1])[-TRUNC if TRUNC is not None else None:]
        return np.hstack([hmm.mf_reverse_cumulative_obs_potentials(self.mf_aBls[state][:t],self,t)\
                [-possible_durations][:,na]
                for state, hmm in enumerate(self.model.HMMs)])

    # TODO TODO the following are only in here for the hard-coded truncation


    def dur_potentials(self,tblock):
        possible_durations = self.segmentlens[tblock:].cumsum()[:TRUNC]
        return self.aDl[possible_durations -1]

    def reverse_dur_potentials(self,tblock):
        possible_durations = rcumsum(self.segmentlens[:tblock+1])[-TRUNC if TRUNC is not None else None:]
        return self.aDl[possible_durations -1]

    def dur_survival_potentials(self,tblock):
        max_dur = self.segmentlens[tblock:].cumsum()[:TRUNC][-1]
        return self.aDsl[max_dur -1]

    def reverse_dur_survival_potentials(self,tblock):
        max_dur = rcumsum(self.segmentlens[:tblock+1])[-TRUNC if TRUNC is not None else None:][0]
        return self.aDsl[max_dur -1]


    def mf_dur_potentials(self,tblock):
        possible_durations = self.segmentlens[tblock:].cumsum()[:TRUNC]
        return self.mf_aDl[possible_durations -1]

    def mf_reverse_dur_potentials(self,tblock):
        possible_durations = rcumsum(self.segmentlens[:tblock+1])[-TRUNC if TRUNC is not None else None:]
        return self.mf_aDl[possible_durations -1]

    def mf_dur_survival_potentials(self,tblock):
        max_dur = self.segmentlens[tblock:].cumsum()[:TRUNC][-1]
        return self.mf_aDsl[max_dur -1]

    def mf_reverse_dur_survival_potentials(self,tblock):
        max_dur = rcumsum(self.segmentlens[:tblock+1])[-TRUNC if TRUNC is not None else None:][0]
        return self.mf_aDsl[max_dur -1]

    ### lots of code copying here, unfortunately TODO

    @property
    def all_expected_stats(self):
        return self.expected_states, self.expected_transcounts, self.expected_durations, \
                self._normalizer, self.subhmm_stats

    @all_expected_stats.setter
    def all_expected_stats(self,vals):
        self.expected_states, self.expected_transcounts, self.expected_durations, \
                self._normalizer, self.subhmm_stats = vals

    def E_step(self):
        # NOTE: this method differs from parent because it passes in self.aBls
        self.clear_caches()
        for hmm in self.model.HMMs:
            assert len(hmm._cache) == 0 and len(hmm._reverse_cache) == 0
        self.all_expected_stats = self._expected_statistics(
                self.trans_potentials, np.log(self.pi_0),
                self.cumulative_obs_potentials, self.reverse_cumulative_obs_potentials,
                self.dur_potentials, self.reverse_dur_potentials,
                self.dur_survival_potentials, self.reverse_dur_survival_potentials,
                self.aBls) # here's the difference
        self.stateseq = self.expected_states.argmax(1) # for plotting

    @line_profiled
    def meanfieldupdate(self):
        # NOTE: this method differs from parent because it passes in self.aBls
        self.clear_caches()
        for hmm in self.model.HMMs:
            assert len(hmm._cache) == 0 and len(hmm._reverse_cache) == 0
        self.all_expected_stats = self._expected_statistics(
                self.mf_trans_potentials, np.log(self.mf_pi_0),
                self.mf_cumulative_obs_potentials, self.mf_reverse_cumulative_obs_potentials,
                self.mf_dur_potentials, self.mf_reverse_dur_potentials,
                self.mf_dur_survival_potentials, self.mf_reverse_dur_survival_potentials,
                self.mf_aBls)
        self.stateseq = self.expected_states.argmax(1) # for plotting

    @line_profiled
    def _expected_statistics(self,
            trans_potentials, initial_state_potential,
            cumulative_obs_potentials, reverse_cumulative_obs_potentials,
            dur_potentials, reverse_dur_potentials,
            dur_survival_potentials, reverse_dur_survival_potentials,
            aBls):
        # NOTE: this method differs from parent because it gets self.aBls
        alphal, alphastarl, _ = hsmm_messages_forwards_log(
                trans_potentials,
                initial_state_potential,
                reverse_cumulative_obs_potentials,
                reverse_dur_potentials,
                reverse_dur_survival_potentials,
                np.empty((self.T,self.num_states)),np.empty((self.T,self.num_states)))

        betal, betastarl, normalizer = hsmm_messages_backwards_log(
                trans_potentials,
                initial_state_potential,
                cumulative_obs_potentials,
                dur_potentials,
                dur_survival_potentials,
                np.empty((self.T,self.num_states)),np.empty((self.T,self.num_states)),
                right_censoring=False)

        expected_states = self._expected_states(
                alphal, betal, alphastarl, betastarl, normalizer)

        expected_transitions = self._expected_transitions(
                alphal, betastarl, trans_potentials, normalizer) # TODO assumes homog trans

        expected_durations = self._expected_durations(
                dur_potentials,cumulative_obs_potentials,
                alphastarl, betal, normalizer)

        ### here's the different bit!
        # also compute subhmm expected stats, using aBls

        subhmm_expected_states = [np.zeros((self.Tfull,hmm.num_states)) for hmm in self.model.HMMs]
        subhmm_expected_trans = [np.zeros((hmm.num_states,hmm.num_states)) for hmm in self.model.HMMs]

        for tblock in xrange(self.Tblock):
            for tblockend, obs, dur in zip(
                    xrange(tblock,min(self.Tblock,tblock+TRUNC)),
                    cumulative_obs_potentials(tblock)[0],dur_potentials(tblock)):

                tstart = self.segmentstarts[tblock]
                tend = self.segmentstarts[tblockend] + self.segmentlens[tblockend]

                weights = np.exp(alphastarl[tblock] + betal[tblockend] + obs + dur - normalizer)
                if tblockend == self.Tblock-1:
                    weights += np.exp(
                            alphastarl[tblock] + betal[tblockend] + obs +
                            dur_survival_potentials(tblockend) - normalizer)

                for state, (hmm, weight) in enumerate(zip(self.model.HMMs,weights)):
                    if weight > 0:
                        states, trans, _ = hmm.mf_expected_statistics( # NOTE: calls mf version!
                                aBls[state][tstart:tend],self,tstart,tend) # here's where aBls are used
                        subhmm_expected_states[state][tstart:tend] += weight*states
                        subhmm_expected_trans[state] += weight*trans

        subhmm_stats = [[states, trans, self.data]
                for states, trans in zip(subhmm_expected_states,subhmm_expected_trans)]

        return expected_states, expected_transitions, expected_durations, normalizer, subhmm_stats


    def _expected_durations(self,
            dur_potentials,cumulative_obs_potentials,
            alphastarl,betal,normalizer):
        logpmfs = -np.inf*np.ones((self.Tfull,alphastarl.shape[1]))
        errs = np.seterr(invalid='ignore') # logaddexp(-inf,-inf)
        # TODO censoring not handled correctly here
        for tblock in xrange(self.Tblock):
            possible_durations = self.segmentlens[tblock:].cumsum()[:TRUNC]
            obs_potentials, _ = cumulative_obs_potentials(tblock)
            logpmfs[possible_durations -1] = np.logaddexp(
                    dur_potentials(tblock) + alphastarl[tblock]
                    + betal[tblock:tblock+TRUNC if TRUNC is not None else None]
                    + obs_potentials - normalizer,
                    logpmfs[possible_durations -1])
        np.seterr(**errs)
        return np.exp(logpmfs.T)


    ### OLD

    # def block_cumulative_likelihoods(self,startblock,stopblock,possible_durations):
    #     # could recompute possible_durations given startblock, stopblock,
    #     # trunc/truncblock, and self.segmentlens, but why redo that effort?
    #     return np.vstack([self.block_cumulative_likelihood_state(startblock,stopblock,state,possible_durations) for state in range(self.num_states)]).T

    # keep this one for forawrd sampling. maybe reimplement it?
    def block_cumulative_likelihood_state(self,startblock,stopblock,state,possible_durations):
        start = self.segmentstarts[startblock]
        stop = self.segmentstarts[stopblock] if stopblock < len(self.segmentstarts) else None
        return self.model.HMMs[state].cumulative_obs_potentials(self.aBls[state][start:])\
                [possible_durations -1]
        # return np.logaddexp.reduce(self.model.HMMs[state].messages_forwards(self.aBls[state][start:stop])[possible_durations-1],axis=1)


