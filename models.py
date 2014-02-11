from __future__ import division
import numpy as np
from matplotlib import cm

from pyhsmm.models import HMM, HSMM, WeakLimitHDPHMM, WeakLimitHDPHSMM, \
        WeakLimitHDPHSMMPossibleChangepoints

import states

# TODO constrained SubHMM
# TODO SVI

class SubWeakLimitHDPHMM(WeakLimitHDPHMM):
    def messages_forwards(self,aBl):
        return self._states_class._messages_forwards_log(
                self.trans_distn.trans_matrix,
                self.init_state_distn.pi_0,
                aBl)

    def get_aBl(self,data):
        self.add_data(
                data=data,
                stateseq=np.zeros(data.shape[0]), # dummy
                )
        return self.states_list.pop().aBl

class WeakLimitHDPHSMMSubHMMs(WeakLimitHDPHSMM):
    _states_class = states.HSMMSubHMMStates
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

        super(WeakLimitHDPHSMMSubHMMs,self).__init__(obs_distns=self.HMMs,**kwargs)

    def resample_obs_distns(self):
        for hmm in self.HMMs:
            # NOTE: don't need to resample subHMM states here because they are
            # resampled all at once with the superstates
            hmm.resample_parameters()

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

class WeakLimitHDPHSMMSubHMMsPossibleChangepoints(
        WeakLimitHDPHSMMSubHMMs, WeakLimitHDPHSMMPossibleChangepoints):
    _states_class = states.HSMMSubHMMStatesPossibleChangepoints
