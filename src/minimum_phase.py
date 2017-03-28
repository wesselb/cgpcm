import numpy as np

import core.plot as plot
import core.data as data


f, k, h = data.load_hrir()
h_mp = h.minimum_phase()

p = plot.Plotter2D()
p.subplot(2, 2, 1)
p.title('Filter')
p.plot(h, label='Original')
p.plot(h_mp, label='Minimum phase')
p.subplot(2, 2, 2)
p.title('Amplitude of Spectrum')
p.plot(h.x, np.log(np.abs(np.fft.fft(h.y))))
p.plot(h_mp.x, np.log(np.abs(np.fft.fft(h_mp.y))))
p.subplot(2, 1, 2)
p.title('Argument of Spectrum')
p.plot(h.x, np.angle(np.fft.fft(h.y)))
p.plot(h_mp.x, np.angle(np.fft.fft(h_mp.y)))

p.show()
