#!/usr/bin/env python
import argparse

from core.cgpcm import AKM
from core.plot import Plotter2D
from core.util import *
from core.tfutil import *
from core.data import Data
import core.out as out


def plot(p, k, psd):
    mu, lowers, uppers = k
    for lower, upper in zip(lowers, uppers):
        p.fill(lower.x, lower.y, upper.y, fill_alpha=.01, fill_colour='b')
    p.plot(mu.x, mu.y, line_width=1.5, line_colour='b')
    p.hide_ticks(x=True)
    if psd:
        p.lims(x=(-.05, .05), y=(-20, 30))
    else:
        p.lims(x=(-.5, .5), y=(-.25, 2))


parser = argparse.ArgumentParser(description='Plot prior kernel or PSD.')
parser.add_argument('--psd', action='store_true', help='plot PSD')
parser.add_argument('--show', action='store_true', help='show plots')
args = parser.parse_args()

# Initialise
sess = Session()
seed()
recipe_c = {'sess': sess,
            'e': Data(np.linspace(0, 1, 150), None),
            'nx': 0,
            'nh': 51,
            'k_len': .1,
            'k_wiggles': 1,
            'causal': True}
recipe_ac = dict2(recipe_c, causal=False)

# Construct models
mod_c = AKM.from_recipe(**recipe_c)
mod_ac = AKM.from_recipe(**recipe_ac)

# Sample
out.section('sampling')
tk = np.linspace(-.6, .6, 301)
out.section('causal model')
k_c = mod_c.k_prior(tk, granularity=.75, iters=5000, psd=args.psd)
out.section_end()
out.section('acausal model')
k_ac = mod_ac.k_prior(tk, granularity=.75, iters=5000, psd=args.psd)
out.section_end()
out.section_end()

# Plot stuff
out.section('plotting')
p = Plotter2D(figure_size=(5, 2), grid_style='none')
p.subplot(1, 2, 1)
plot(p, k_ac, args.psd)
p.subplot(1, 2, 2)
plot(p, k_c, args.psd)
p.save('output/prior_{}.pdf'.format('psd' if args.psd else 'kernel'))
if args.show:
    p.show()
out.section_end()
