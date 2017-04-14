import controller as c
import numpy as np
from core.tf_util import *
from core.plot import Plotter2D

p = Plotter2D()
s = c.start_session()
t = c.fetch([['timit']], remote=True)[0]
m = c.load_trained_model(s, t, [])
x = np.linspace(-0.1, 0.1, 1000)

# windows = ['none', 'hamming', 'boxcar', 'blackman']
# psd_h_mean = m.predict_h(x).mean.autocorrelation().fft_db()
#
# for i, name in enumerate(windows):
#     p.subplot(1, len(windows), i + 1)
#     p.title(name)
#     p.plot(psd_h_mean, label='Averaged')
#     for j in range(5):
#         p.plot(m.predict_h(x, samples_h=1).mean.window(name).autocorrelation().fft_db(),
#                label='Single sample ({})'.format(j))
#     p.lims(x=(0, 1e3))
#     p.show_legend()
#
# p.show()

p.figure()
p.title('Posterior covariance')
C = s.run(mul3(m.Kh, m.h.var, m.Kh))
p.plt.imshow(C)
p.plt.colorbar()
p.show()
