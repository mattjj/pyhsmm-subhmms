from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

import pyhsmm
from pyhsmm.util.stats import cov
from pyhsmm.util.text import progprint_xrange, progprint
from pyhsmm.util.general import sgd_manypass, minibatchsize, rle

import models

Nsuper = 4
Nsub = 3
T = 2000
obs_dim = 2

try:
    import brewer2mpl
    plt.set_cmap(brewer2mpl.get_map('Set1','qualitative',max(3,min(8,Nsuper))).mpl_colormap)
except:
    pass

obs_hypparams = dict(
        mu_0=np.zeros(obs_dim),
        sigma_0=np.eye(obs_dim),
        kappa_0=0.05,
        nu_0=obs_dim+10,
        )

dur_hypparams = dict(
        r_support=np.array([80,82,84]),r_probs=np.ones(3)/3.,
        alpha_0=10*5,
        beta_0=10*2,
        )

true_obs_distnss = [[pyhsmm.distributions.Gaussian(**obs_hypparams) for substate in xrange(Nsub)]
        for superstate in xrange(Nsuper)]

true_dur_distns = [pyhsmm.distributions.NegativeBinomialIntegerR2Duration(
    **dur_hypparams) for superstate in range(Nsuper)]

truemodel = models.HSMMSubHMMs(
        init_state_concentration=6.,
        sub_init_state_concentration=6.,
        alpha=3.,
        sub_alpha=3.,
        obs_distnss=true_obs_distnss,
        dur_distns=true_dur_distns)

datas, labelss = zip(*[truemodel.generate(T) for itr in xrange(30)])
training_size = minibatchsize(datas)

# only keep the last example around for plotting
truemodel.states_list = truemodel.states_list[-2:]

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
changepointss = []
for data, labels in zip(datas,labelss):
    _, durations = rle(labels)
    temp = np.concatenate(((0,),durations.cumsum()))
    changepoints = zip(temp[:-1],temp[1:])
    changepoints[-1] = (changepoints[-1][0],T) # because last duration might be censored
    changepointss.append(changepoints)

# optionally split changepoints
# changepoints = [pair for (a,b) in changepoints for pair in [(a,a+(b-a)//2), (a+(b-a)//2,b)]]
print len(changepoints)


# # plot things to check!!
# plt.figure()
# for idx, (labels, changepoints) in enumerate(zip(labelss,changepointss)):
#     plt.subplot(len(changepointss),1,idx)
#     plt.plot(labels)
#     plt.vlines([start for start,stop in changepoints],0,labels.max(),color='r',linestyle='--')
# plt.show()

model = models.HSMMSubHMMsPossibleChangepoints(
        init_state_concentration=6.,
        sub_init_state_concentration=6.,
        alpha=Nsuper,
        sub_alpha=Nsub,
        obs_distnss=obs_distnss,
        dur_distns=dur_distns)

###############
#  inference  #
###############


### sampling for initialization

# for data, changepoints in zip(datas[-2:], changepointss[-2:]):
#     model.add_data(data,changepoints=changepoints)

# for itr in progprint_xrange(25):
#     model.resample_model()

# plt.figure()
# model.plot()
# plt.gcf().suptitle('sampled')

# model.states_list = []

# svi

scores = []
for t, ((data,changepoints), rho_t) in progprint(
        sgd_manypass(0,0.6,zip(*[datas,changepointss]),npasses=1)):
    model.meanfield_sgdstep(
            data, minibatchsize(data) / training_size, rho_t,
            changepoints=changepoints)

# decode the last two just to take a look
for data, changepoints in zip(*[datas[-2:],changepointss[-2:]]):
    model.add_data(data,changepoints=changepoints)
    s = model.states_list[-1]
    s.mf_Viterbi()

plt.figure()
model.plot()
plt.gcf().suptitle('fit')

plt.matshow(np.vstack(
    (
        np.tile(truemodel.states_list[0].stateseq,(500,1)),
        np.tile(model.states_list[0].stateseq,(500,1)),
    )))
plt.vlines([start for start, _ in changepointss[-2]],0,1000,color='k',linestyle='--')

# # model.resample_model()
# # s = model.states_list[0]
# # s.E_step()
# # s.meanfieldupdate()

# # plt.figure()
# # plt.plot(sum(stats[0].sum(1) for stats in s.subhmm_stats))

plt.show()

