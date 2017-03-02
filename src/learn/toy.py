from tensorflow.contrib.opt.python.training.external_optimizer import \
    ScipyOptimizerInterface as SciPyOpt
import pickle

from core.cgpcm import AKM, VCGPCM
from core.plot import Plotter2D
from core.tfutil import *


def normalise_energy(x, dt, causal):
    single = False
    if not (type(x) == list or type(x) == tuple):
        x = [x]
        single = True
    if causal:
        energy = np.sum(x[0][(len(x[0]) - 1) / 2:] ** 2 * dt)
    else:
        energy = np.sum(x[0] ** 2 * dt)
    return x[0] / energy ** .5 if single \
        else map(lambda y: y / energy ** .5, x)


def normalise(x):
    if type(x) == list or type(x) == tuple:
        return map(lambda y: y / max(x[0]), x)
    else:
        return x / max(x)


sess = SessionDecoratorDebug(tf.Session())

# Causal sample:
# Seed: 214748361 + 118
# Noise: 0
# Noise init: 0.01 ** 2
# Redraw noise: 3
# Sample alpha: kernel_length
# Sample gamma: .8 * kernel_length

# Acausal sample:
# Seed: 21474836 + 202
# Noise: 0.15 ** 2
# Noise init: 0.15 ** 2
# Redraw noise: 0
# Sample alpha: .8 * kernel_length
# Sample gamma: .8 * kernel_length

seed = 21474836 + 202
tf.set_random_seed(seed)
np.random.seed(seed)

do_train = True
do_train_fn = 'approx_q'
do_plot = True
do_plot_fn = 'known_kernels_acs_acm'

if do_train:
    print 'Loading data...'
    n = 300
    choose = 300
    nh = 51
    nx = 80
    nk = 501
    k_len = .3
    noise_init = 0.15 ** 2
    gamma_init = length_scale(.6 * k_len)
    omega_init = length_scale(2. / nx)
    k_stretch = 3

    kernel_len = .5 * k_len
    noise = 0.15 ** 2

    train_iters_pre = 1000
    train_iters = 0
    ess_burn = 500
    ess_samps = 200

    test_kernel = False
    test_sample = False
    redraw_noise = 0
    init_at_optimal_mean = False

    causal = False
    causal_sample = False

    pars = {'alpha': length_scale(k_len),
            'gamma': var_pos(to_float(gamma_init), 'gamma'),
            'omega': var_pos(to_float(omega_init), 'omega'),
            'K': 1, 'M': 1, 'N': 1,
            's2': var_pos(to_float(noise_init), 's2'),
            's2_f': var_pos(to_float(1.), 's2_f')}
    initialise_uninitialised_variables(sess)

    t = np.linspace(-1, 1, n)[:, None]
    th = np.linspace(-k_stretch * k_len, k_stretch * k_len, nh)[:, None]
    tx = np.linspace(-1, 1, nx)[:, None]
    tk = np.linspace(-k_stretch * 2 * k_len, k_stretch * 2 * k_len, nk)[:,
         None]

    pars_akm = {'alpha': length_scale(.8 * kernel_len),
                'gamma': length_scale(.8 * kernel_len),
                'omega': length_scale(2. / nx),
                's2': to_float(noise),
                's2_f': to_float(.1)}
    akm = AKM(pars_akm,
              tf.constant(th[:, 0]),
              tf.constant(tx[:, 0]),
              t[:, 0],
              None,
              sess)
    akm.generate_sample(t, tk)

    # Redraw noise
    h_samp = akm.sample['h']
    for i in range(redraw_noise):
        akm.generate_sample(t, tk)
    akm.sample['h'] = h_samp

    if test_kernel:
        k = akm.k(causal=causal_sample)
        p = Plotter2D()
        p.plot(tk, k)
        p.show()

    k_true, y, _ = akm.f(causal=causal_sample)
    k_true, y = normalise(k_true[:, None]), y[:, None]
    h_true = normalise_energy(akm.h(tf.constant(tk)),
                              dt=tk[1, 0] - tk[0, 0], causal=causal_sample)
    y_noisy = y + np.random.randn(n, 1) * np.sqrt(noise)

    if test_sample:
        p = Plotter2D()
        p.plot(y)
        p.plot(y_noisy)
        p.show()

    choice = np.random.choice(n, choose, replace=False)
    y_sample = y_noisy[choice, 0]
    t_sample = t[choice, 0]

    print 'Constructing model...'
    mod = VCGPCM(pars,
                 tf.constant(th[:, 0]),
                 tf.constant(tx[:, 0]),
                 t_sample,
                 y_sample,
                 sess,
                 causal=causal,
                 causal_id=False)
    initialise_uninitialised_variables(sess)

    if init_at_optimal_mean:
        tf.assign(mod.h.mean, sess.run(mul3(mod.iKh, akm.Kh, h_samp)))

    if train_iters_pre == 0 and train_iters == 0:
        mod.precompute()

    if train_iters_pre > 0:
        print 'Pretraining...'
        mod.precompute()
        opt = SciPyOpt(-mod.elbo(),
                       options={'disp': True, 'maxiter': train_iters_pre,
                                'maxls': 10},
                       var_list=map(get_var, ['s2_f', 'muh', 'Sh', 's2']))
        initialise_uninitialised_variables(sess)
        opt.minimize(sess)

    if train_iters > 0:
        mod.undo_precompute()
        opt = SciPyOpt(-mod.elbo(),
                       options={'disp': True, 'maxiter': train_iters,
                                'maxls': 10},
                       var_list=map(get_var,
                                    ['s2_f', 'muh', 'Sh', 's2', 'gamma']))
        initialise_uninitialised_variables(sess)
        opt.minimize(sess)
        mod.precompute()

    print 'Predicting'
    print '  function using MF...'
    f_mf = mod.predict_f(tf.constant(t))
    print '  kernel using MF...'
    k_mf = normalise(mod.predict_k(tf.constant(tk)))
    print '  PSD using MF...'
    PSD_mf = mod.predict_k(tf.constant(tk), psd=True)
    print '  filter using MF...'
    h_mf = normalise_energy(mod.predict_h(tf.constant(tk)),
                            dt=tk[1, 0] - tk[0, 0], causal=causal)

    # Sample from SMF using ESS
    samples = mod.sample(ess_samps, burn=ess_burn, display=True)

    print 'Predicting'
    print '  function using SMF...'
    f_smf = mod.predict_f(tf.constant(t), smf=True, samples_h=samples)
    print '  kernel using SMF...'
    k_smf = normalise(mod.predict_k(tf.constant(tk), samples_h=samples))
    print '  PSD using SMF...'
    PSD_smf = mod.predict_k(tf.constant(tk), samples_h=samples, psd=True)
    print '  filter using SMF...'
    h_smf = normalise_energy(mod.predict_h(tf.constant(tk), samples_h=samples),
                             dt=tk[1, 0] - tk[0, 0], causal=causal)

    print 'Saving...'
    result = {'mf': {'f': f_mf,
                     'k': k_mf,
                     'psd': PSD_mf,
                     'h': h_mf},
              'smf': {'f': f_smf,
                      'k': k_smf,
                      'psd': PSD_smf,
                      'h': h_smf},
              's2': sess.run(mod.s2),
              's2_f': sess.run(mod.s2_f),
              'alpha': mod.pars['alpha'],
              'gamma': sess.run(mod.pars['gamma']),
              'h_samp': sess.run(mul(akm.Kh, akm.sample['h'])),
              'h_true': h_true,
              'k_true': k_true,
              't': t, 'n': n,
              'tk': tk, 'nk': nk,
              'th': th, 'nh': nh,
              'tx': tx, 'nx': nx,
              'choose': choose,
              'init_at_optimal_mean': init_at_optimal_mean,
              'samples': samples,
              'seed': seed,
              'redraw_noise': redraw_noise,
              'y': y, 'y_noisy': y_noisy,
              't_sample': t_sample, 'y_sample': y_sample,
              'causal': causal, 'causal_sample': causal_sample,
              'k_stretch': k_stretch, 'k_len': k_len,
              'noise': noise}

    with open('models/' + do_train_fn, 'w') as f:
        pickle.dump(result, f)

if do_plot_fn:
    print 'Loading...'
    with open('models/' + do_plot_fn) as f:
        result = pickle.load(f)

    t, y = result['t'], result['y']
    nh, nk, nx = result['nh'], result['nk'], result['nx']
    th, tk, tx = result['th'], result['tk'], result['tx']
    t_sample, y_sample = result['t_sample'], result['y_sample']
    f_mf, f_smf = result['mf']['f'], result['smf']['f']
    k_mf, k_smf = result['mf']['k'], result['smf']['k']
    h_mf, h_smf = result['mf']['h'], result['smf']['h']
    PSD_mf, PSD_smf = result['mf']['psd'], result['smf']['psd']
    k_stretch, k_len = result['k_stretch'], result['k_len']
    h_true, k_true = result['h_true'], result['k_true']
    causal, causal_sample = result['causal'], result['causal_sample']
    s2 = result['s2']

    print 'Plotting...'
    p = Plotter2D()
    p.figure('Result', fig_size=(10, 2))

    p.subplot2grid((1, 5), (0, 0), colspan=2)
    p.plot(tx, 0 * tx, ma_st='o', ma_co='k', ma_si=2, li_st='none')
    p.plot(t_sample, y_sample, la='Sample', li_st='None', ma_st='o',
           ma_co='g', ma_si=2)
    print 'Function'
    for (f, f_lower, f_upper, f_std), la, co in [(f_mf, 'MF', 'b'),
                                                 (f_smf, 'SMF', 'cyan')]:
        p.fill(t[:, 0], f_lower, f_upper, fil_co=co)
        p.plot(t[:, 0], f, li_co=co, la='Learned ({})'.format(la), li_wi=1.5)
        print 'SMSE {}: {:.4f}'.format(la, smse(f, y))
        print 'MLL {}: {:.4f}'.format(la, mll(f, f_std, y))
    p.plot(t, y, la='Truth', li_st='None', ma_st='o',
           ma_co='r', ma_si=2)
    p.plt.axis('tight')
    p.hide(x=True, y=True)

    p.subplot2grid((1, 5), (0, 2))
    p.plt.axvline(x=0, linewidth=1, color='k')
    p.plot(th, 0 * th, ma_st='o', ma_co='k', ma_si=2, li_st='none')
    print 'Filter'
    for (h, h_lower, h_upper, h_std), la, co in [(h_mf, 'MF', 'b'),
                                                 (h_smf, 'SMF', 'cyan')]:
        p.fill(tk[:, 0], h_lower, h_upper, fil_co=co)
        p.plot(tk[:, 0], h, la='Learned ({})'.format(la), li_co=co, li_wi=1.5)
        print 'SMSE {}: {:.4f}'.format(la, smse(h, h_true))
        print 'MLL {}: {:.4f}'.format(la, mll(h, h_std, h_true))
    if h_true is not None:
        p.plot(tk, normalise_energy(h_true,
                                    tk[1] - tk[0], causal_sample), la='Truth',
               li_co='r', li_wi=1.5)
    p.plt.axis('tight')
    p.hide(x=True, y=True)
    p.lims(x=(-k_stretch * k_len, k_stretch * k_len))

    p.subplot2grid((1, 5), (0, 3))
    p.plt.axvline(x=0, linewidth=1, color='k')
    p.plot(th, 0 * th, ma_st='o', ma_co='k', ma_si=2, li_st='none')
    ry = np.correlate(np.squeeze(y), np.squeeze(y), mode='full')
    max_lag = t[-1] - t[0]
    p.plot(np.linspace(-max_lag, max_lag, len(ry)), ry / max(ry),
           la='Autocorrelation', li_co='g', li_wi=1.5)
    print 'Kernel'
    for (k, k_lower, k_upper, k_std), la, co in [(k_mf, 'MF', 'b'),
                                                 (k_smf, 'SMF', 'cyan')]:
        p.fill(tk[:, 0], k_lower, k_upper, fil_co=co)
        p.plot(tk[:, 0], k, la='Learned ({})'.format(la), li_co=co, li_wi=1.5)
        print 'SMSE {}: {:.4f}'.format(la, smse(k, k_true))
        print 'MLL {}: {:.4f}'.format(la, mll(k, k_std, k_true))
    if k_true is not None:
        p.plot(tk, k_true, la='Truth', li_co='r', li_wi=1.5)
    p.plt.axis('tight')
    p.hide(x=True, y=True)
    p.lims(x=(0, 2.5 * k_len))

    p.subplot2grid((1, 5), (0, 4))
    freqs = fft_freq(nk, tk[1, 0] - tk[0, 0])
    psd_emp = psd(zero_pad(ry / max(ry) / len(ry), 1000))
    p.plot(fft_freq(len(psd_emp), t[1, 0] - t[0, 0]), psd_emp,
           la='Periodogram',
           li_co='g', li_wi=1.5)
    for (PSD, PSD_lower, PSD_upper, PSD_std), la, co in [(PSD_mf, 'MF', 'b'),
                                                         (PSD_smf, 'SMF',
                                                          'cyan')]:
        p.fill(freqs, PSD_lower, PSD_upper, fil_co=co)
        p.plot(freqs, PSD, la='Learned ({})'.format(la), li_co=co, li_wi=1.5)
    if k_true is not None:
        psd_true = psd(np.squeeze(k_true / max(k_true) / len(k_true)))
        p.plot(freqs, psd_true, la='Truth', li_co='r', li_wi=1.5)
    p.plt.axis('tight')
    p.hide(x=True, y=True)
    p.lims(x=(0, 20), y=(-35, -5))

    p.save('output/learning_known_kernels_{}_{}.pdf'.format(
        'causal_sample' if causal_sample else 'acausal_sample',
        'cgpcm' if causal else 'gpcm'
    ))
    p.show()
