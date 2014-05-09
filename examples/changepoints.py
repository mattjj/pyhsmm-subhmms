from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

import pyhsmm
from pyhsmm.util.stats import cov
from pyhsmm.util.text import progprint_xrange

import models

Nsuper = 4
Nsub = 3
T = 2500
obs_dim = 2

try:
    import brewer2mpl
    plt.set_cmap(brewer2mpl.get_map('Set1','qualitative',max(3,min(8,Nsuper))).mpl_colormap)
except:
    pass

obs_hypparams = dict(
        mu_0=np.zeros(obs_dim),
        sigma_0=np.eye(obs_dim),
        kappa_0=0.1,
        nu_0=obs_dim+10,
        )

dur_hypparams = dict(
        r_support=np.array([40,42,44]),r_probs=np.ones(3)/3.,
        alpha_0=10*10,
        beta_0=10*2,
        )

true_obs_distnss = [[pyhsmm.distributions.Gaussian(**obs_hypparams) for substate in xrange(Nsub)]
        for superstate in xrange(Nsuper)]

true_dur_distns = [pyhsmm.distributions.NegativeBinomialIntegerR2Duration(
    **dur_hypparams) for superstate in range(Nsuper)]

truemodel = models.HSMMSubHMMs(
        init_state_concentration=6.,
        sub_init_state_concentration=6.,
        alpha=10.,
        sub_alpha=10.,
        obs_distnss=true_obs_distnss,
        dur_distns=true_dur_distns)

data, _ = truemodel.generate(T)

truemodel.plot()
plt.gcf().suptitle('truth')


##################
#  set up model  #
##################

Nmaxsuper = 2*Nsuper
Nmaxsub = 2*Nsub

obs_distnss = \
        [[pyhsmm.distributions.Gaussian(**obs_hypparams)
            for substate in range(Nmaxsub)] for superstate in range(Nmaxsuper)]

dur_distns = \
        [pyhsmm.distributions.NegativeBinomialIntegerR2Duration(
            **dur_hypparams) for superstate in range(Nmaxsuper)]

# !!! cheat to get the changepoints !!! #
temp = np.concatenate(((0,),truemodel.states_list[0].durations.cumsum()))
changepoints = zip(temp[:-1],temp[1:])
changepoints[-1] = (changepoints[-1][0],T) # because last duration might be censored

# split changepoints
# changepoints = [pair for (a,b) in changepoints for pair in [(a,a+(b-a)//2), (a+(b-a)//2,b)]]
print len(changepoints)

model = models.HSMMSubHMMsPossibleChangepoints(
        init_state_concentration=6.,
        sub_init_state_concentration=6.,
        alpha=6.,
        sub_alpha=6,
        obs_distnss=obs_distnss,
        dur_distns=dur_distns)

model.add_data(data=data,changepoints=changepoints)

###############
#  inference  #
###############


# NEXT:
# SVI

plt.figure()
model.plot()
plt.gcf().suptitle('sampled')

scores = []
for itr in progprint_xrange(5):
    scores.append(model.meanfield_coordinate_descent_step())

# i think sampling isn't really moving it far, since the vlb keeps improving
model._reregister_state_sequences()
for itr in progprint_xrange(25):
    model.resample_model()

for itr in progprint_xrange(5):
    scores.append(model.meanfield_coordinate_descent_step())

plt.figure()
model.plot()
plt.gcf().suptitle('fit')

plt.figure()
plt.plot(scores)


# plt.show()

plt.matshow(np.vstack(
    (
        np.tile(truemodel.states_list[0].stateseq,(1000,1)),
        np.tile(model.states_list[0].stateseq,(1000,1)),
    )))

# # model.resample_model()
# # s = model.states_list[0]
# # s.E_step()
# # s.meanfieldupdate()

# # plt.figure()
# # plt.plot(sum(stats[0].sum(1) for stats in s.subhmm_stats))

plt.show()

