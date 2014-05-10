from __future__ import division
import numpy as np
from matplotlib import cm

from pyhsmm.models import HMM, HSMM, WeakLimitHDPHMM, WeakLimitHDPHSMM, \
        WeakLimitHDPHSMMPossibleChangepoints, HSMMPossibleChangepoints

import states

class Dummy(object):
    pass

# TODO caching!
# key could be t or tblock. let's do t

class SubHMM(HMM):
    def __init__(self,*args,**kwargs):
        super(SubHMM,self).__init__(*args,**kwargs)
        self._clear_message_caches()

    def _clear_message_caches(self):
        self._cache = {}
        self._reverse_cache = {}


    def get_aBl(self,data):
        self.add_data(data=data,stateseq=np.zeros(data.shape[0]))
        return self.states_list.pop().aBl

    def get_mf_aBl(self,data):
        self.add_data(data=data,stateseq=np.zeros(data.shape[0]))
        return self.states_list.pop().mf_aBl


    def cumulative_obs_potentials(self,aBl,t=None):
        if t not in self._cache or t is None:
            self._cache[t] = self._states_class._messages_forwards_log(
                    self.trans_distn.trans_matrix,self.init_state_distn.pi_0,aBl)
        alphal = self._cache[t]
        return np.logaddexp.reduce(alphal,axis=1)

    def reverse_cumulative_obs_potentials(self,aBl,t=None):
        if t not in self._reverse_cache or t is None:
            self._reverse_cache[t] = self._states_class._messages_backwards_log(
                    self.trans_distn.trans_matrix,aBl)
        betal = self._reverse_cache[t]
        return np.logaddexp.reduce(betal + np.log(self.init_state_distn.pi_0) + aBl,axis=1)

    def mf_cumulative_obs_potentials(self,mf_aBl,t=None):
        if t not in self._cache or t is None:
            self._cache[t] = self._states_class._messages_forwards_log(
                    self.trans_distn.exp_expected_log_trans_matrix,
                    self.init_state_distn.exp_expected_log_init_state_distn,
                    mf_aBl)
        mf_alphal = self._cache[t]
        return np.logaddexp.reduce(mf_alphal,axis=1)

    def mf_reverse_cumulative_obs_potentials(self,mf_aBl,t=None):
        if t not in self._reverse_cache or t is None:
            self._reverse_cache[t] = self._states_class._messages_backwards_log(
                    self.trans_distn.exp_expected_log_trans_matrix,mf_aBl)
        mf_betal = self._reverse_cache[t]
        return np.logaddexp.reduce(
                mf_betal +
                np.log(self.init_state_distn.exp_expected_log_init_state_distn) +
                mf_aBl,axis=1)

    def mf_expected_statistics(self,mf_aBl,tstart=None,tend=None):
        if tstart is not None and tstart in self._cache:
            mf_alphal = self._cache[tstart][:tend-tstart]
        else:
            mf_alphal = self._states_class._messages_forwards_log(
                    self.trans_distn.exp_expected_log_trans_matrix,
                    self.init_state_distn.exp_expected_log_init_state_distn,
                    mf_aBl)

        if tend is not None and tend in self._reverse_cache:
            mf_betal = self._reverse_cache[tend][-(tend-tstart):]
        else:
            mf_betal = self._states_class._messages_backwards_log(
                    self.trans_distn.exp_expected_log_trans_matrix,mf_aBl)

        return self._states_class._expected_statistics_from_messages(
                self.trans_distn.exp_expected_log_trans_matrix,
                mf_aBl,mf_alphal,mf_betal)


    def get_vlb(self):
        # NOTE: no states term b/c the HSMM states normalizer takes care of that
        vlb = 0.
        vlb += self.trans_distn.get_vlb()
        vlb += self.init_state_distn.get_vlb()
        vlb += sum(o.get_vlb() for o in self.obs_distns)
        return vlb

    def meanfield_update_from_stats(self,statslist):
        old_states = self.states_list
        self.states_list = []
        for mf_expected_states, mf_expected_transcounts, data in statslist:
            dummy = Dummy()
            dummy.mf_expected_states, dummy.mf_expected_transcounts, dummy.data = \
                    mf_expected_states, mf_expected_transcounts, data
            self.states_list.append(dummy)

        self.meanfield_update_parameters()

        self.states_list = old_states

    def meanfield_sgdstep_from_stats(self,stateslist,minibatchfrac,stepsize):
        mb_states_list = []
        for mf_expected_states, mf_expected_transcounts, data in stateslist:
            dummy = Dummy()
            dummy.mf_expected_states, dummy.mf_expected_transcounts, dummy.data = \
                    mf_expected_states, mf_expected_transcounts, data
            mb_states_list.append(dummy)

        self.meanfield_sgdstep_parameters(mb_states_list,minibatchfrac,stepsize)


class SubWeakLimitHDPHMM(SubHMM,WeakLimitHDPHMM):
    pass


class HSMMSubHMMs(HSMM):
    _states_class = states.HSMMSubHMMStates
    _subhmm_class = SubHMM

    def __init__(self,
            obs_distnss=None,
            subHMMs=None,
            sub_alpha=None,
            sub_alpha_a_0=None,sub_alpha_b_0=None,
            sub_init_state_concentration=None,
            **kwargs):
        self.obs_distnss = obs_distnss
        if subHMMs is None:
            assert obs_distnss is not None
            self.HMMs = [
                    self._subhmm_class(
                        obs_distns=obs_distns,
                        alpha=sub_alpha,
                        alpha_a_0=sub_alpha_a_0,alpha_b_0=sub_alpha_b_0,
                        init_state_concentration=sub_init_state_concentration,
                        )
                    for obs_distns in obs_distnss]
        else:
            self.HMMs = subHMMs

        HSMM.__init__(self,obs_distns=self.HMMs,**kwargs)

    def resample_obs_distns(self):
        for hmm in self.HMMs:
            # NOTE: don't need to resample subHMM states here because they are
            # resampled all at once with the superstates
            hmm.resample_parameters()

    def meanfield_update_obs_distns(self):
        for state, hmm in enumerate(self.HMMs):
            hmm.meanfield_update_from_stats(
                [s.subhmm_stats[state] for s in self.states_list])

    def meanfield_sgdstep_obs_distns(self,mb_states_list,minibatchfrac,stepsize):
        for state, hmm in enumerate(self.HMMs):
            hmm.meanfield_sgdstep_from_stats(
                [s.subhmm_stats[state] for s in mb_states_list],
                minibatchfrac,stepsize)

    def plot_observations(self,colors=None,states_objs=None):
        # NOTE: colors are superstate colors
        if colors is None:
            colors = self._get_colors()
        if states_objs is None:
            states_objs = self.states_list

        cmap = cm.get_cmap()
        used_superstates = self._get_used_states(states_objs)
        for superstate,hmm in enumerate(self.HMMs):
            if superstate in used_superstates:
                substates = hmm._get_used_states()
                num_substates = len(substates)
                hmm.plot_observations(
                        colors=dict(
                            (substate,colors[superstate]+offset)
                            for substate,offset in zip(substates,
                                np.linspace(-0.5,0.5,num_substates,endpoint=True)/12.5)))

    def _reregister_state_sequences(self):
        for hmm in self.HMMs:
            hmm.states_list = []

        for s in self.states_list:
            s.substates_list = []
            indices = np.concatenate(((0,),np.cumsum(s.durations_censored[:-1])))
            for state, startidx, dur in zip(s.stateseq_norep,indices,s.durations_censored):
                self.HMMs[state].add_data(
                        s.data[startidx:startidx+dur],
                        stateseq=s.subhmm_stats[state][0][startidx:startidx+dur].argmax(1))
                s.substates_list.append(self.HMMs[state].states_list[-1])


class WeakLimitHDPHSMMSubHMMs(HSMMSubHMMs,WeakLimitHDPHSMM):
    _subhmm_class = SubWeakLimitHDPHMM
    def __init__(self,
            obs_distnss=None,
            subHMMs=None,
            sub_alpha=None,sub_gamma=None,
            sub_alpha_a_0=None,sub_alpha_b_0=None,sub_gamma_a_0=None,sub_gamma_b_0=None,
            sub_init_state_concentration=None,
            **kwargs):
        self.obs_distnss = obs_distnss
        if subHMMs is None:
            assert obs_distnss is not None
            self.HMMs = [
                    self._subhmm_class(
                        obs_distns=obs_distns,
                        alpha=sub_alpha,gamma=sub_gamma,
                        alpha_a_0=sub_alpha_a_0,alpha_b_0=sub_alpha_b_0,
                        gamma_a_0=sub_gamma_a_0,gamma_b_0=sub_gamma_b_0,
                        init_state_concentration=sub_init_state_concentration,
                        )
                    for obs_distns in obs_distnss]
        else:
            self.HMMs = subHMMs

        WeakLimitHDPHSMM.__init__(self,obs_distns=self.HMMs,**kwargs)


class HSMMSubHMMsPossibleChangepoints(HSMMSubHMMs, HSMMPossibleChangepoints):
    _states_class = states.HSMMSubHMMStatesPossibleChangepoints

class WeakLimitHDPHSMMSubHMMsPossibleChangepoints(
        WeakLimitHDPHSMMSubHMMs, WeakLimitHDPHSMMPossibleChangepoints):
    _states_class = states.HSMMSubHMMStatesPossibleChangepoints

