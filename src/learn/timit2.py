from tensorflow.contrib.opt.python.training.external_optimizer import \
    ScipyOptimizerInterface as SciPyOpt
import pickle
import scipy.io as sio

from core.gpcm import VGPCM
from core.plotter import Plotter2D
from core.utils import *


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

seed = 21474836
tf.set_random_seed(seed)
np.random.seed(seed)

cgpcm = True
do_train = True
do_plot = True

if do_train:
    print 'Loading data...'
    data_size = 800  # Fragment length
    n = 300  # Data under consideration
    choose = 200  # Number of training points
    imp_region = (150, 200)
    nh = 80
    nx = 120
    nk = 501
    k_len = .4
    noise_init = 0.001 ** 2
    gamma_init = length_scale(.4 * k_len)
    omega_init = length_scale(2. / nx)
    k_stretch = 2

    train_iters_pre = 100
    train_iters = 0
    ess_burn = 100
    ess_samps = 100

    noise = 0

    causal = cgpcm

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

    audio = sio.loadmat('audioWS.mat')['audio']
    margin = 200
    y = audio[11499 + margin:11499 + margin + data_size].squeeze()
    y = (y - np.mean(y)) / np.std(y, ddof=1)  # Normalise
    t = np.linspace(-1, 1, len(y))[:, None]
    choice = sorted(np.random.choice(len(y), n, replace=False))
    y_full, t_full = y, t
    y, t = y[choice], t[choice]
    k_true = None

    y_noisy = y + np.random.randn(n) * np.sqrt(noise)

    y_sample = np.concatenate((y_noisy[:imp_region[0]], y_noisy[imp_region[1] - 1:]))
    t_sample = np.concatenate((t[:imp_region[0]], t[imp_region[1] - 1:]))
    choice = sorted(np.random.choice(len(y_sample), choose, replace=False))
    y_sample = y_sample[choice]
    t_sample = t_sample[choice]

    # Determine free indices
    free_inds = np.array(list(set(xrange(len(y_sample))) - set(choice)))
    free_inds[free_inds >= imp_region[0]] += imp_region[1] - imp_region[0]

    print 'Constructing model...'
    mod = VGPCM(pars,
                tf.constant(th[:, 0]),
                tf.constant(tx[:, 0]),
                t_sample,
                y_sample,
                sess,
                causal=causal,
                causal_id=False)
    initialise_uninitialised_variables(sess)

    if train_iters_pre == 0 and train_iters == 0:
        mod.precompute()

    if train_iters_pre > 0:
        print 'Pretraining...'
        mod.precompute()
        opt = SciPyOpt(-mod.elbo(),
                       options={'disp': True,
                                'maxiter': train_iters_pre,
                                'maxls': 10},
                       var_list=map(get_var, ['s2_f', 'muh', 'Sh', 's2']))
        initialise_uninitialised_variables(sess)
        opt.minimize(sess)

    if train_iters > 0:
        print 'Training...'
        mod.undo_precompute()
        opt = SciPyOpt(-mod.elbo(),
                       options={'disp': True,
                                'maxiter': train_iters,
                                'maxls': 10},
                       var_list=map(get_var, ['s2_f',
                                              'muh',
                                              'Sh',
                                              's2',
                                              'gamma']))
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
    # f_smf, k_smf, h_smf, PSD_smf, samples = None, None, None, None, None

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
              'k_true': k_true,
              't': t, 'n': n,
              'tk': tk, 'nk': nk,
              'th': th, 'nh': nh,
              'tx': tx, 'nx': nx,
              'choose': choose,
              'samples': samples,
              'seed': seed,
              'y': y, 'y_noisy': y_noisy,
              't_sample': t_sample, 'y_sample': y_sample,
              'causal': causal,
              'k_stretch': k_stretch, 'k_len': k_len,
              'noise': noise,
              'y_full': y_full, 't_full': t_full,
              'free_inds': free_inds,
              'imp_region': imp_region}

    with open('models/timit_{}'.format('cgpcm' if cgpcm else 'gpcm'),
              'w') as f:
        pickle.dump(result, f)

if do_plot:
    print 'Loading...'
    with open('models/timit_{}'.format('cgpcm' if cgpcm else 'gpcm')) as f:
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
    causal = result['causal']
    y_full, t_full = result['y_full'], result['t_full']
    imp_region = result['imp_region']

    print 'Stats...'


    def judge(y, f, inds):
        rmse = np.mean((f[0][inds] - y[inds]) ** 2) ** .5
        print 'RMSE: {:.3f}'.format(rmse)
        s2 = result['s2'] * f[3][inds]
        msll = np.mean(.5 * np.log(2 * np.pi * s2)
                       + .5 * (f[0][inds] - y[inds]) ** 2 / s2)
        print 'MSLL: {:.3f}'.format(msll)


    print 'MF'
    print 'Extrapolation:'
    judge(y, f_mf, np.arange(*imp_region))
    print 'Interpolation:'
    judge(y, f_mf, result['free_inds'])

    print 'SMF'
    print 'Extrapolation:'
    judge(y, f_smf, np.arange(*imp_region))
    print 'Interpolation:'
    judge(y, f_smf, result['free_inds'])

    print 'Plotting...'
    p = Plotter2D()
    p.figure('Result', fig_size=(25, 12))

    p.subplot(2, 2, 1)
    p.plot(tx, 0 * tx, ma_st='o', ma_co='k', ma_si=2, li_st='none')
    p.plot(t, y, la='Truth', li_st='None', ma_st='o',
           ma_co='r', ma_si=3)
    p.plot(t_sample, y_sample, la='Sample', li_st='None', ma_st='o',
           ma_co='g', ma_si=3)
    for (f, f_lower, f_upper, _), la, co in [(f_mf, 'MF', 'b'),
                                             (f_smf, 'SMF', 'cyan')]:
        p.fill(t[:, 0], f_lower, f_upper, fil_co=co)
        p.plot(t[:, 0], f, li_co=co, la='Learned ({})'.format(la), li_wi=1.5)
        print 'RMSE {}: {:.4f}'.format(la, np.mean((f - y) ** 2) ** .5)
    p.plt.axvline(x=t[imp_region[0]], color='k', linewidth=1)
    p.plt.axvline(x=t[imp_region[1] - 1], color='k', linewidth=1)
    p.plt.axis('tight')
    p.hide(x=True, y=True)

    p.subplot(2, 2, 2)
    p.plt.axvline(x=0, linewidth=1, color='k')
    p.plot(th, 0 * th, ma_st='o', ma_co='k', ma_si=2, li_st='none')
    for (h, h_lower, h_upper, _), la, co in [(h_mf, 'MF', 'b'),
                                          (h_smf, 'SMF', 'cyan')]:
        p.fill(tk[:, 0], h_lower, h_upper, fil_co=co)
        p.plot(tk[:, 0], h, la='Learned ({})'.format(la), li_co=co, li_wi=1.5)
    p.plt.axis('tight')
    p.hide(x=True, y=True)
    p.lims(x=(-k_stretch * k_len, k_stretch * k_len))

    p.subplot(2, 2, 3)
    p.plt.axvline(x=0, linewidth=1, color='k')
    p.plot(th, 0 * th, ma_st='o', ma_co='k', ma_si=2, li_st='none')
    ry = np.correlate(np.squeeze(y_full), np.squeeze(y_full), mode='full')
    p.plot(np.linspace(t_full[0] - t_full[-1],
                       t_full[-1] - t_full[0], 2 * len(t_full) - 1),
           ry / max(ry),
           la='Autocorrelation', li_co='g', li_wi=1.5)
    for (k, k_lower, k_upper, _), la, co in [(k_mf, 'MF', 'b'),
                                          (k_smf, 'SMF', 'cyan')]:
        p.fill(tk[:, 0], k_lower, k_upper, fil_co=co)
        p.plot(tk[:, 0], k, la='Learned ({})'.format(la), li_co=co, li_wi=1.5)
    p.plt.axis('tight')
    p.hide(x=True, y=True)
    p.lims(x=(0, 2.5 * k_len))

    p.subplot(2, 2, 4)
    freqs = fft_freq(nk, tk[1, 0] - tk[0, 0])
    psd_emp = psd(zero_pad(ry / max(ry) / len(ry), 1000))
    p.plot(fft_freq(len(psd_emp), t_full[1] - t_full[0]), psd_emp,
           la='Periodogram',
           li_co='g', li_wi=1.5)
    for (PSD, PSD_lower, PSD_upper, _), la, co in [(PSD_mf, 'MF', 'b'),
                                                (PSD_smf, 'SMF', 'cyan')]:
        p.fill(freqs, PSD_lower, PSD_upper, fil_co=co)
        p.plot(freqs, PSD, la='Learned ({})'.format(la), li_co=co, li_wi=1.5)
    p.plt.axis('tight')
    p.lims(x=(0, 80), y=(-65, 0))
    p.hide(x=True, y=True)

    p.save('output/timit_{}.pdf'.format('cgpcm' if cgpcm else 'gpcm'))
    p.show()
