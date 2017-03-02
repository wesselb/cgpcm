from core.tfutil import *


with tf.Session() as sess:
    # Some complicated expression to test more advanced gradients
    v = ones(3)
    fac = randn([3]).eval()
    div = to_float(5)
    rho = sum(v * fac) / div

    x = tf.constant(np.array([[0.1, 0.2], [0.2, 0.3]]))
    y = bvn_cdf2(x, rho)

    delta = 1e-8

    print 'y1 wrt x11'
    e = tf.constant(np.array([[1., 0], [0, 0]])) * delta / 2.
    y1 = bvn_cdf2(x - e, rho)
    y2 = bvn_cdf2(x + e, rho)
    print (y2[0].eval() - y1[0].eval()) / delta
    print tf.gradients([y[0]], x)[0].eval()[0, 0]

    print 'y2 wrt x21'
    e = tf.constant(np.array([[0, 0], [1., 0]])) * delta / 2.
    y1 = bvn_cdf2(x - e, rho)
    y2 = bvn_cdf2(x + e, rho)
    print (y2[1].eval() - y1[1].eval()) / delta
    print tf.gradients([y[1]], x)[0].eval()[1, 0]

    print 'y1 wrt rho'
    e = to_float(1) * delta / 2
    y1 = bvn_cdf2(x, rho - e)
    y2 = bvn_cdf2(x, rho + e)
    print (y2[0].eval() - y1[0].eval()) / delta
    print tf.gradients([y[0]], rho)[0].eval()

    print 'y2 wrt v1'
    e = tf.constant(np.array([1., 0, 0])) * delta / 2
    rho1 = sum((v - e) * fac) / div
    rho2 = sum((v + e) * fac) / div
    y1 = bvn_cdf2(x, rho1)
    y2 = bvn_cdf2(x, rho2)
    print (y2[1].eval() - y1[1].eval()) / delta
    print tf.gradients([y[1]], v)[0].eval()[0]

    print 'y2 wrt v3'
    e = tf.constant(np.array([0, 0, 1.])) * delta / 2
    rho1 = sum((v - e) * fac) / div
    rho2 = sum((v + e) * fac) / div
    y1 = bvn_cdf2(x, rho1)
    y2 = bvn_cdf2(x, rho2)
    print (y2[1].eval() - y1[1].eval()) / delta
    print tf.gradients([y[1]], v)[0].eval()[2]

