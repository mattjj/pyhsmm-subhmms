from __future__ import division

from pyhsmm.internals.states import HSMMStatesPython, HSMMStatesPossibleChangepoints

# TODO should be able to initialize substateseqs

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
        # NOTE: already generated in HMMs, so this is a little weird
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

    def clear_caches(self):
        self._aBls = None
        super(HSMMSubHMMStates,self).clear_caches()

    def resample(self,temp=None):
        self._remove_substates_from_subHMMs()
        super(HSMMSubHMMStates,self).resample() # resamples superstates
        self._resample_substates()

    def cumulative_likelihood_state(self,start,stop,state):
        return np.logaddexp.reduce(self.model.HMMs[state].messages_forwards(self.aBls[state][start:stop]),axis=1)

    def cumulative_likelihoods(self,start,stop):
        return np.hstack([self.cumulative_likelihood_state(start,stop,state)[:,na]
            for state in range(self.state_dim)])

    def likelihood_block_state(self,start,stop,state):
        return np.logaddexp.reduce(self.model.HMMs[state].messages_forwards(self.aBls[state][start:stop])[-1])

    def likelihood_block(self,start,stop):
        return np.array([self.likelihood_block_state(start,stop,state)
            for state in range(self.state_dim)])

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

class HSMMSubHMMStatesPossibleChangepoints(HSMMSubHMMStates,HSMMStatesPossibleChangepoints):
    def messages_backwards(self,*args,**kwargs):
        return HSMMStatesPossibleChangepoints.messages_backwards(self,*args,**kwargs)

    def sample_forwards(self,betal,betastarl):
        return HSMMStatesPossibleChangepoints.sample_forwards(self,betal,betastarl)

    def block_cumulative_likelihoods(self,startblock,stopblock,possible_durations):
        # could recompute possible_durations given startblock, stopblock,
        # trunc/truncblock, and self.segmentlens, but why redo that effort?
        return np.vstack([self.block_cumulative_likelihood_state(startblock,stopblock,state,possible_durations) for state in range(self.state_dim)]).T

    def block_cumulative_likelihood_state(self,startblock,stopblock,state,possible_durations):
        start = self.segmentstarts[startblock]
        stop = self.segmentstarts[stopblock] if stopblock < len(self.segmentstarts) else None
        return np.logaddexp.reduce(self.model.HMMs[state].messages_forwards(self.aBls[state][start:stop])[possible_durations-1],axis=1)

    def generate(self):
        # TODO override generate someday
        raise NotImplementedError

