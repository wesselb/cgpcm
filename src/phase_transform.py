import numpy as np

import core.plot as plot
import core.data as data


f, k, h = data.load_hrir()
h_mp = h.minimum_phase()
h_zp = h.zero_phase()

p = plot.Plotter2D()
p.subplot(2, 2, 1)
p.title('Filter')
p.plot(h, label='Original')
p.plot(h_mp, label='Minimum phase')
p.plot(h_zp, label='Zero phase')
p.show_legend()

p.subplot(2, 2, 2)
p.title('Amplitude of Spectrum')
p.plot(np.log(np.abs(np.fft.fft(h.y))))
p.plot(np.log(np.abs(np.fft.fft(h_mp.y))))
p.plot(np.log(np.abs(np.fft.fft(h_zp.y))))

p.subplot(2, 1, 2)
p.title('Argument of Spectrum')
p.plot(np.angle(np.fft.fft(h.y)))
p.plot(np.angle(np.fft.fft(h_mp.y)))
p.plot(np.angle(np.fft.fft(np.fft.fftshift(h_zp.y))))


p.show()
