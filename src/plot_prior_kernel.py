from core.cgpcm import AKM
from core.plot import Plotter2D
from core.util import *
from core.tfutil import *


def plot(p, k):
    mu, lowers, uppers = k
    for lower, upper in zip(lowers, uppers):
        p.fill(tk, lower, upper, fill_alpha=.01, fill_colour='b')
    p.plot(tk, mu, line_width=1.5, line_colour='b')
    p.hide_ticks(x=True, y=True)
    p.lims(x=(-.5, .5), y=(-.25, 2))


sess = Session()
seed()

pars_conf_c = {'sess': sess,
               't': np.linspace(0, 1, 150),
               'y': [],
               'nx': 0,
               'nh': 51,
               'k_len': .1,
               'k_wiggles': 1,
               'causal': True}
pars_conf_ac = dict2(pars_conf_c, causal=False)

mod_c = AKM.from_pars_config(**pars_conf_c)
mod_ac = AKM.from_pars_config(**pars_conf_ac)

print 'Sampling kernels...'
tk = np.linspace(-.6, .6, 301)
k_c = mod_c.k_prior(tk, granularity=.75, iters=5000)
k_ac = mod_ac.k_prior(tk, granularity=.75, iters=5000)

p = Plotter2D(figure_size=(5, 2), grid_style='none')
p.subplot(1, 2, 1)
plot(p, k_ac)
p.subplot(1, 2, 2)
plot(p, k_c)

p.save('output/prior_kernel.pdf')
p.show()
