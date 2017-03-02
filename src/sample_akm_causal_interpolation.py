from core.cgpcm import AKM
from core.plot import Plotter2D
from core.util import *
from core.kernel import DEQ
from core.dist import Normal

config.reg = 1e-6

sess = tf.Session()

# 1212

seed = 99998 # 2147483647
tf.set_random_seed(seed)
np.random.seed(seed)


def length_scale(ls):
    return .5 / ls ** 2


k_len = .4
gamma_frac = .5
n = 450
nh = 31
nx = 100  # Irrelevant
noise_init = 1e-3  # Irrelevant
noise = 0
k_stretch = 2
pars = {'alpha': length_scale(k_len),
        'gamma': var_pos(to_float(length_scale(k_len * gamma_frac)), 'gamma'),
        'omega': length_scale(2. / nx),
        'K': 1, 'M': 1, 'N': 1,
        's2': var_pos(to_float(noise_init), 's2'),
        's2_f': var_pos(to_float(1.), 's2_f')}
sess.run(tf.global_variables_initializer())

t = np.linspace(-1, 1, n)
th = np.linspace(-k_stretch * k_len, k_stretch * k_len, nh)
tx = np.linspace(-1, 1, nx)
tk = np.linspace(-k_stretch * 2 * k_len, k_stretch * 2 * k_len, 501)
tk_pos = np.linspace(0, k_stretch * 2 * k_len, 501)

print 'Constructing model...'
mod = AKM(pars,
          tf.constant(th),
          tf.constant(tx),
          t,
          None,
          sess)


# Compute filters
def h_conditioned(h0):
    deq = DEQ({'alpha': pars['alpha'],
               'gamma': pars['gamma'],
               's2': 1.})
    thu = np.array([h0])
    Kh = deq(th[:, None])
    Ku = deq(thu[:, None])
    Kuh = deq(thu[:, None], th[:, None])
    A = trisolve(tf.cholesky(reg(Ku)), Kuh)
    Kh = reg(Kh - mul(A, A, adj_a=True))
    h = sess.run(Normal(reg(Kh)).sample())
    h[(nh - 1) / 2] = h0  # Correct for round-off errors
    return h


def interpolate_h(t, h):
    deq = DEQ({'alpha': pars['alpha'],
               'gamma': pars['gamma'],
               's2': 1.})
    Kh = deq(th[:, None])
    Kf = deq(t[:, None])
    Khf = deq(th[:, None], t[:, None])
    A = tf.cholesky_solve(tf.cholesky(reg(Kh)), Khf)
    return sess.run(mul(A, h, adj_a=True))


def h_sample():
    deq = DEQ({'alpha': pars['alpha'],
               'gamma': pars['gamma'],
               's2': 1.})
    Kh = deq(th[:, None])
    return sess.run(Normal(reg(Kh)).sample())



def h_inv(h):
    deq = DEQ({'alpha': pars['alpha'],
               'gamma': pars['gamma'],
               's2': 1.})
    Kh = deq(th[:, None])
    Lh = tf.cholesky(reg(Kh))
    return sess.run(tf.cholesky_solve(Lh, h))

h_from = -h_sample()
h_to = -h_conditioned(0.)

print np.sum(h_from[(nh - 1)/2:] ** 2)
print np.sum(h_to[(nh - 1)/2:] ** 2)

h_from_inv, h_to_inv = h_inv(h_from), h_inv(h_to)

# Compute kernels and samples
num = 5

mod.generate_sample(t, tk)
hs, ks, fs = [], [], []
cs = color_range(num)
for i, frac in enumerate(np.linspace(0, .75, num)):
    print 'Iteration {}/{}, fraction {:.2f}'.format(i + 1, num, frac)
    h_inv = frac * h_from_inv + (1 - frac) * h_to_inv
    h = frac * h_from + (1 - frac) * h_to
    mod.sample['h'] = h_inv
    k, f, K = mod.construct_sample(causal=True)
    ks.append(k), fs.append(f), hs.append(h)

print 'Plotting...'
p = Plotter2D(fig_size=(9, 2.5), font_size=13)
p.figure('Kernels')

for i in range(num):
    p.subplot(3, num, i + 1)
    p.plot(tk_pos, tk_pos * 0, li_co='k')
    p.plot(tk_pos, interpolate_h(tk_pos, hs[i]), li_wi=1)
    p.ax.xaxis.set_visible(False)
    p.ax.yaxis.set_ticklabels([])
    p.ax.yaxis.set_ticks([])
    if i == 0:
        p.labels(y='$h$')

for i in range(num):
    p.subplot(3, num, num + i + 1)
    # p.plt.axvline(x=0, linewidth=1, color='k')
    p.plot(tk, tk * 0, li_co='k')
    p.plot(tk, ks[i], li_wi=1)
    p.ax.xaxis.set_visible(False)
    p.ax.yaxis.set_ticklabels([])
    p.ax.yaxis.set_ticks([])
    if i == 0:
        p.labels(y='$k_{f\,|\,h}$')
    p.lims(x=(-2 * k_len, 2 * k_len))
           # y=(0, .5))

for i in range(num):
    p.subplot(3, num, 2 * num + i + 1)
    p.plot(t, fs[i], li_wi=1)
    p.ax.xaxis.set_visible(False)
    p.ax.yaxis.set_ticklabels([])
    p.ax.yaxis.set_ticks([])
    if i == 0:
        p.labels(y='$f\,|\,h$')

p.save('output/interpolation.pdf')
p.show()
