from core.tfutil import _mvn_cdf, _bvn_cdf
import numpy as np
from time import time


def bench(fn, name, iters=100):
    print 'Benchmark: {}'.format(name)
    start = time()
    for i in range(iters):
        fn()
    dur = time() - start
    print '  Duration:           {:.3f} s'.format(dur)
    print '  Duration/iteration: {:.3f} s'.format(dur / iters)


num = 50000
x = np.random.rand(num, 2)
mu = np.array([[0.2, 0.4]])
var = np.array([[3., 0.1], [0.1, 4.]])

print 'Warming up...'
_mvn_cdf(np.zeros(2), np.zeros(2), np.eye(2))

bench(lambda: _mvn_cdf(x, mu, var), '_mvn_cdf')
bench(lambda: _bvn_cdf(x, mu, var), '_bvn_cdf')





