from core.cgpcm import AKM
from core.plot import Plotter2D
from core.util import *
from core.kernel import DEQ
from core.dist import Normal

sess = tf.Session()

# seed = 2147483647
# tf.set_random_seed(seed)
# np.random.seed(seed)


def length_scale(ls):
    return .5 / ls ** 2


def r2z(r):
    return 2 ** .5 * r / (2 - r ** 2) ** .5


k_len = .2
frac_gamma = .2
n = 200
nh = 200
nx = 100
noise_init = 1e-3
noise = 0
pre_train = True
k_stretch = 2
pars = {'alpha': length_scale(k_len),
        'gamma': var_pos(to_float(length_scale(frac_gamma * k_len)), 'gamma'),
        'omega': length_scale(2. / nx),
        'K': 1, 'M': 1, 'N': 1,
        's2': var_pos(to_float(noise_init), 's2'),
        's2_f': var_pos(to_float(1.), 's2_f')}
sess.run(tf.global_variables_initializer())

t = np.linspace(-1, 1, n)
th = np.linspace(-k_stretch * k_len, k_stretch * k_len, nh)
tx = np.linspace(-1, 1, nx)
tk_stretch = 10
tk = np.linspace(-tk_stretch * k_len, tk_stretch * k_len, 1000)

print 'Constructing model...'
mod = AKM(pars,
          tf.constant(th),
          tf.constant(tx),
          t,
          None,
          sess)

mod.generate_sample(t, tk)
# print 'Acausal'
# k_samp, f_samp, K = mod.construct_sample()
# print 'Causal'
# k_samp_c, f_samp_c, K_c = mod.construct_sample(causal=True)

print 'Acausal'
k_samp = mod.construct_kernel()
print 'Causal'
k_samp_c = mod.construct_kernel(causal=True)


def psd_freqs(t):
    return np.fft.fftshift(np.fft.fftfreq(len(t), t[1] - t[0]))


def psd(f):
    return 10 * np.log(np.abs(np.fft.fftshift(np.fft.fft(f))) ** 2)


def plot_psd(p, t, f):
    p.plot(psd_freqs(t), psd(f))



print 'Plotting...'
p = Plotter2D(fig_size=(20, 12))
p.figure('Result')

p.subplot(1, 2, 1)
p.title('Kernel')
p.plot(tk, k_samp)
p.plot(tk, k_samp_c)
p.lims(x=(-5 * k_len, 5 * k_len))

# p.subplot(1, 3, 2)
# p.title('Function')
# p.plot(t, f_samp)
# p.plot(t, f_samp_c)

p.subplot(1, 2, 2)
p.title('PSD')
plot_psd(p, tk, k_samp)
plot_psd(p, tk, k_samp_c)
max_f = 1 / k_len / frac_gamma
p.lims(x=(-3 * max_f, 3 * max_f))


p.save('output/out.pdf')
p.show()
