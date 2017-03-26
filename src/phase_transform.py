import numpy as np

import core.plot as plot
import core.data as data


f, k, h = data.load_hrir(1000)

print 'GO'
h = h.zero_pad(1000).minimum_phase()
print 'Finished'

p = plot.Plotter2D()
p.subplot(2, 2, 1)
p.title('Filter')
p.plot(h)
p.subplot(2, 2, 2)
p.title('log $|F|$')
p.plot(h.x, np.log(np.abs(np.fft.fft(h.y))))
p.subplot(2, 2, 3)
p.title('arg $F$')
p.plot(h.x, np.angle(np.fft.fft(h.y)))


p.show()
