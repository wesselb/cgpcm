import sys

from core.tfutil import *
from core.util import *
from core.plot import Plotter2D
import core.out as out
import core.data
import core.cgpcm
import core.learn


def plot(p, x, x_pred, x_noisy=None, inducing_points=None):
    mu, lower, upper, std = x_pred
    p.plot(x.t, x.y, label='Truth', line_colour='r')
    if x_noisy is not None:
        p.plot(x_noisy.t, x_noisy.y,
               label='Observed',
               line_style='none',
               marker_style='o',
               marker_colour='g',
               marker_size=3)
    p.fill(x.t, lower, upper, fill_colour='b')
    p.plot(x.t, mu, label='Learned', line_colour='b')
    if inducing_points is not None:
        p.plot(inducing_points, inducing_points * 0,
               line_style='none',
               marker_style='o',
               marker_colour='k',
               marker_size=3)
    p.show_legend()


args = sys.argv[1:]

sess = Session()
seed(48)

causal = '--causal' in args
causal_mod = '--causal-model' in args
resample = 1
n = 150
nx = 50
nh = 31
k_len = 0.15
k_wiggles = 1
noise = 1e-4
iters_pre = 400
iters = 2000
samps = 500

out.section('loading data')
f, k, h, pars = core.data.load_akm(sess=sess,
                                   n=n,
                                   nh=nh,
                                   k_len=.5 * k_len,
                                   k_wiggles=k_wiggles,
                                   causal=causal,
                                   resample=resample)
f_noisy = f.make_noisy(noise)
out.section_end()

# Plotter2D().plot(k.t, k.y).show()
# exit()

mod = core.cgpcm.VCGPCM.from_pars_config(sess=sess,
                                         t=f_noisy.t,
                                         y=f_noisy.y,
                                         nx=nx,
                                         nh=nh,
                                         k_len=k_len,
                                         k_wiggles=2 * k_wiggles,
                                         causal=causal_mod,
                                         name_vars=True)

# Precomputation
out.section('precomputation')
mod.precompute()
out.section_end()

# Train MF
out.section('training MF')
elbo = mod.elbo()
fetches_config = [{'name': 'ELBO', 'tensor': elbo, 'modifier': '.2e'},
                  {'name': 's2', 'tensor': mod.s2, 'modifier': '.2e'},
                  {'name': 's2_f', 'tensor': mod.s2_f, 'modifier': '.2e'},
                  {'name': 'gamma', 'tensor': mod.gamma, 'modifier': '.2e'}]
core.learn.minimise_lbfgs(sess, -elbo,
                          vars=map(get_var, ['muh', 'Sh']),
                          iters=iters_pre,
                          fetches_config=fetches_config)
core.learn.minimise_lbfgs(sess, -elbo,
                          vars=map(get_var, ['s2', 's2_f', 'muh', 'Sh']),
                          iters=iters,
                          fetches_config=fetches_config)
out.section_end()

# Predict MF
out.section('predicting MF')
f_pred = mod.predict_f(f.t)
k_pred = mod.predict_k(k.t)
pos_i = nearest_index(h.t, .2 * k_len)
h_pred = mod.predict_h(h.t,
                       assert_positive_at_index=pos_i)
out.section_end()

# Train SMF
out.section('training SMF')
samples = mod.sample(iters=samps)
out.section_end()

# Predict SMF
out.section('predicting SMF')
f_pred_smf = mod.predict_f(f.t, samples_h=samples)
k_pred_smf = mod.predict_k(k.t, samples_h=samples)
h_pred_smf = mod.predict_h(h.t,
                           samples_h=samples,
                           assert_positive_at_index=pos_i)
out.section_end()

# Plotting
out.section('plotting')
p = Plotter2D(figure_size=(20, 10))

p.subplot(2, 3, 1)
p.title('Filter')
th = sess.run(mod.th)
plot(p, h, h_pred, inducing_points=th)

p.subplot(2, 3, 2)
p.title('Kernel')
plot(p, k, k_pred, inducing_points=th)

p.subplot(2, 3, 3)
p.title('Function')
tx = sess.run(mod.tx)
plot(p, f, f_pred, x_noisy=f_noisy, inducing_points=tx)

p.subplot(2, 3, 4)
p.title('Filter')
plot(p, h, h_pred_smf, inducing_points=th)

p.subplot(2, 3, 5)
p.title('Kernel')
plot(p, k, k_pred_smf, inducing_points=th)

p.subplot(2, 3, 6)
p.title('Function')
plot(p, f, f_pred_smf, x_noisy=f_noisy, inducing_points=tx)

qualifiers = '{}_sample'.format('causal' if causal else 'acausal')
qualifiers += '_{}'.format('cgpcm' if causal_mod else 'gpcm')
p.save('output/out_{}.pdf'.format(qualifiers))

out.section_end()
