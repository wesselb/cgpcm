from core.cgpcm import AKM
from core.plot import Plotter2D
from core.util import *
from core.kernel import DEQ
from core.dist import Normal

sess = tf.Session()

seed = 2147483647
tf.set_random_seed(seed)
np.random.seed(seed)


def length_scale(ls):
    return .5 / ls ** 2


def r2z(r):
    return 2 ** .5 * r / (2 - r ** 2) ** .5


k_len = .5
n = 150
nh = 31
nx = 100
noise_init = 1e-3
noise = 0
deq_len = k_len * .3
pre_train = True
k_stretch = 2
pars = {'alpha': length_scale(k_len),
        'gamma': var_pos(to_float(length_scale(k_len * r2z(.3))), 'gamma'),
        'omega': length_scale(2. / nx),
        'K': 1, 'M': 1, 'N': 1,
        's2': var_pos(to_float(noise_init), 's2'),
        's2_f': var_pos(to_float(1.), 's2_f')}
sess.run(tf.global_variables_initializer())

t = np.linspace(-1, 1, n)
th = np.linspace(-k_stretch * k_len, k_stretch * k_len, nh)
tx = np.linspace(-1, 1, nx)
tk = np.linspace(-k_stretch * 2 * k_len, k_stretch * 2 * k_len, 500)

deq = DEQ({'alpha': 0.,
           'gamma': length_scale(deq_len),
           's2': 1.})
y = sess.run(Normal(reg(deq(t[:, None]))).sample())
k_true = sess.run(deq(tk[:, None], tk[:1][:, None]))

y_noisy = y + np.random.randn(n, 1) * np.sqrt(noise)

print 'Constructing model...'
mod = AKM(pars,
          tf.constant(th),
          tf.constant(tx),
          t,
          y_noisy,
          sess)

print 'Sampling kernels...'
k_mu, k_lower, k_upper = mod.prior_kernel(tk, iters=5000, causal=False)
k_mu_c, k_lower_c, k_upper_c = mod.prior_kernel(tk, causal=True, iters=5000)


p = Plotter2D(fig_size=(5, 2))

p.subplot(1, 2, 1)
for lower, upper in zip(k_lower, k_upper):
    p.fill(tk, lower, upper, fil_al=.04, fil_co='b')
p.plot(tk, k_mu, li_wi=1.5, li_co='b')
p.hide(x=True, y=True)
p.lims(x=(-1, 1), y=(-.5, 1.5))

p.subplot(1, 2, 2)
for lower, upper in zip(k_lower_c, k_upper_c):
    p.fill(tk, lower, upper, fil_al=.04, fil_co='b')
p.plot(tk, k_mu_c, li_wi=1.5, li_co='b')
p.hide(x=True, y=True)
p.plt.axis('tight')
p.lims(x=(-1, 1), y=(-0.25, 0.75))
p.save('output/prior.pdf')
p.show()

exit()

p.subplot(1, 2, 1)
for lower, upper in zip(k_lower, k_upper):
    p.fill(tk, lower, upper, fil_al=.04, fil_co='b')
p.plot(tk, k_mu, li_wi=1.5, li_co='b')
p.hide(x=True)
p.ax.grid(False, which='major')
p.lims(x=(-.5, .5), y=(20 - 40, 20))

p.subplot(1, 2, 2)
for lower, upper in zip(k_lower_c, k_upper_c):
    p.fill(tk, lower, upper, fil_al=.04, fil_co='b')
p.plot(tk, k_mu_c, li_wi=1.5, li_co='b')
p.hide(x=True)
p.ax.grid(False, which='major')
p.plt.axis('tight')
p.lims(x=(-.5, .5), y=(20 - 40, 20))
p.save('output/prior_psd.pdf')
p.show()





exit()

print 'Sampling functions...'
k_samps, f_samps = [], []
k_samps_c, f_samps_c = [], []

cs = color_range(2)
for i in range(len(cs)):
    print '  Iteration {}/{}'.format(i + 1, len(cs))
    mod.generate_sample(t, tk)
    print '    Acausal...'
    k_samp, f_samp, K = mod.construct_sample()
    print '    Causal...'
    k_samp_c, f_samp_c, K_c = mod.construct_sample(causal=True)
    k_samps.append(k_samp)
    f_samps.append(f_samp)
    k_samps_c.append(k_samp_c)
    f_samps_c.append(f_samp_c)

print 'Plotting...'
p = Plotter2D(fig_size=(20, 12))
p.figure('Result')

p.subplot(2, 2, 1)
p.title('Kernel (Acausal)')
p.fill(tk, k_lower, k_upper, fil_co='k')
p.plot(tk, k_mu, li_co='k')
for c, k_samp in zip(cs, k_samps):
    p.plot(tk, k_samp, li_co=c)

# p.subplot(2, 3, 2)
# p.title('Kernel Matrix (Acausal)')
# p.plt.imshow(K)

p.subplot(2, 2, 2)
p.title('Function (Acausal)')
for c, f_samp in zip(cs + ['b'], f_samps):
    p.plot(t, f_samp, li_co=c)

p.subplot(2, 2, 3)
p.title('Kernel (Causal)')
p.fill(tk, k_lower_c, k_upper_c, fil_co='k')
p.plot(tk, k_mu_c, li_co='k')
for c, k_samp in zip(cs, k_samps_c):
    p.plot(tk, k_samp, li_co=c)

# p.subplot(2, 3, 5)
# p.title('Kernel Matrix (Causal)')
# p.plt.imshow(K_c)

p.subplot(2, 2, 4)
p.title('Function (Causal)')
for c, f_samp in zip(cs + ['b'], f_samps_c):
    p.plot(t, f_samp, li_co=c)

p.save('output/out.pdf')
p.show()
