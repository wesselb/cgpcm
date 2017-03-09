#!/usr/bin/env python
import pickle
import argparse

from core.cgpcm import AKM
from core.plot import Plotter2D
from core.kernel import DEQ
from core.dist import Normal
from core.tfutil import *
from core.util import *
from core.data import Data
import core.out as out
import core.learn as learn


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


parser = argparse.ArgumentParser(description='Plot interpolation of filters.')
parser.add_argument('--compute', action='store_true',
                    help='compute interpolation')
parser.add_argument('--show', action='store_true', help='show plots')
args = parser.parse_args()

# Config
fn_cache = 'output/cache/interpolation_data.pickle'
sess = Session()
seed(28)

# Construct model and kernel
mod = AKM.from_recipe(sess=sess,
                      e=Data(np.linspace(0, 1, 150), None),
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
fracs = np.linspace(0, .6, 5)
num_fracs = len(fracs)

if args.compute:
    mod.sample_f(t)
    hs, ks, fs = [], [], []

    progress = learn.Progress(name='computing filters, kernels, and samples',
                              iters=num_fracs,
                              fetches_config=[{'name': 'fraction',
                                               'modifier': '.2f'}])
    for i, frac in enumerate(fracs):
        progress([frac])

        h = (1 - frac) * h_from + frac * h_to
        mod.sample_h(h_inv(h.y[:, None]))

        ks.append(mod.k(tk).y)
        fs.append(mod.f().y)
        hs.append(mod.h(th, assert_positive_at_index=50).y)
    with open(fn_cache, 'w') as f:
        pickle.dump((ks, fs, hs), f)
else:
    with open(fn_cache) as f:
        ks, fs, hs = pickle.load(f)

# Plotting
out.section('plotting')
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
if args.show:
    p.show()
out.section_end()
