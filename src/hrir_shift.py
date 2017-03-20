from core.tf_util import *
import core.util as util
import core.data as data
import core.kernel as kernel
import core.distribution as dist
import core.learn as learn
import core.out as out


sess = Session()
delta_t = tf.Variable(to_float(0))
f, k, h = data.load_hrir()

# Construct kernel
tau_w = 1e-3
tau_f = .2e-3
alpha = util.length_scale(tau_w)
gamma = util.length_scale(tau_f) - .5 * alpha
alpha, alpha_var = var_pos(to_float(alpha))
gamma, gamma_var = var_pos(to_float(gamma))
deq = kernel.DEQ(s2=1., alpha=alpha, gamma=gamma)

# Construct prior
prior = dist.Normal(reg(deq(h.x + delta_t)))

# Optimise log-likelihood
ll = prior.log_pdf(h.y[None, :])
initialise_uninitialised_variables(sess)
learn.minimise_lbfgs(sess, -ll, iters=100,
                     vars=[delta_t],
                     fetches_config=[{'name': 'log-likelihood',
                                      'tensor': ll,
                                      'modifier': '.2e'}])

# Report
out.section('results')
out.kv('alpha', sess.run(alpha), mod='.2e')
out.kv('gamma', sess.run(gamma), mod='.2e')
out.kv('shift', sess.run(1000 * delta_t), unit='ms', mod='.4f')
out.section_end()
