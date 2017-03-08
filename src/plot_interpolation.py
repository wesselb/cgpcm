from core.cgpcm import AKM
from core.plot import Plotter2D
from core.kernel import DEQ
from core.dist import Normal
from core.tfutil import *
from core.util import *
from core.data import Data

import pickle


def h_sample():
    K = kernel(mod.th)
    return sess.run(Normal(reg(K)).sample())


def h_sample_conditioned(h0):
    tu = constant(np.array([0]))
    u = constant(np.array([[h0]]))
    Kh = kernel(mod.th)
    Lu = tf.cholesky(reg(kernel(tu)))
    Kuh = kernel(tu, mod.th)

    # Compute posterior distribution
    A = trisolve(Lu, Kuh)
    K = reg(Kh - mul(A, A, adj_a=True))
    mu = mul(A, trisolve(Lu, u), adj_a=True)
    return sess.run(Normal(K, mu).sample())


def h_posterior_mean(t, h):
    Lh = tf.cholesky(reg(kernel(mod.th)))
    Kuh = kernel(t, mod.th)
    return mul(Kuh, tf.cholesky_solve(Lh, h))


def h_inv(h):
    K = kernel(mod.th)
    L = tf.cholesky(reg(K))
    return sess.run(tf.cholesky_solve(L, h))


sess = Session()
seed(25)

compute = False

mod = AKM.from_pars_config(sess=sess,
                           t=np.linspace(0, 1, 150),
                           y=[],
                           nx=0,
                           nh=51,
                           k_len=.1,
                           k_wiggles=1,
                           causal=True)

kernel = DEQ(s2=1., alpha=mod.alpha, gamma=mod.gamma)

# Sample and normalise to unity energy
th = sess.run(mod.th)
h_from = Data(th, h_sample_conditioned(0)[:, 0])
h_from /= h_from.energy_causal ** .5
h_to = Data(th, h_sample()[:, 0])
h_to /= h_to.energy_causal ** .5

# Compute filter, kernels, and samples
t = np.linspace(0, 1, 250)
th = np.linspace(0, .6, 301)
tk = np.linspace(-.6, .6, 301)

mod.sample_f(t)
hs, ks, fs = [], [], []

fracs = np.array([0, .2, .32, .42, .5])
num_fracs = len(fracs)

if compute:
    for i, frac in enumerate(fracs):
        print 'Iteration {}/{}, fraction {:.2f}'.format(i + 1, num_fracs, frac)
        h = (1 - frac) * h_from + frac * h_to
        mod.sample_h(h_inv(h.y[:, None]))
        ks.append(mod.k(tk))
        fs.append(mod.f())
        hs.append(mod.h(th, assert_positive_at_index=120))
    with open('output/interpolation_data.pickle', 'w') as f:
        pickle.dump((ks, fs, hs), f)
else:
    with open('output/interpolation_data.pickle') as f:
        ks, fs, hs = pickle.load(f)

print 'Plotting...'
p = Plotter2D(figure_size=(9, 2.5), font_size=13)
p.figure('Kernels')

for i in range(num_fracs):
    p.subplot(3, num_fracs, i + 1)
    p.plot(th, th * 0, line_colour='k')
    p.plot(th, hs[i], line_width=1.5)
    p.hide_ticks(x=True, y=True)
    if i == 0:
        p.labels(y='$h$')

for i in range(num_fracs):
    p.subplot(3, num_fracs, num_fracs + i + 1)
    p.plot(tk, tk * 0, line_colour='k')
    p.plot(tk, ks[i], line_width=1.5)
    p.hide_ticks(x=True, y=True)
    if i == 0:
        p.labels(y='$k_{f\,|\,h}$')

for i in range(num_fracs):
    p.subplot(3, num_fracs, 2 * num_fracs + i + 1)
    p.plot(t, fs[i], line_width=1.5)
    p.hide_ticks(x=True, y=True)
    if i == 0:
        p.labels(y='$f\,|\,h$')

p.save('output/interpolation.pdf')
p.show()
