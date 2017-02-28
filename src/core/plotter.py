from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import abc
import warnings
import itertools
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import matplotlib.pyplot as plt


def to_rbga(color):
    """ Convert `color` to a tuple `(red, blue, green, alpha)`. """
    return mpl.colors.colorConverter.to_rgba(color, alpha=1.0)


# Mapping for values from strings
val_map = {'True': True, 'False': False}


def dict_map(source, mapping):
    """
    Replace values in dictionary `mapping` with their values in `source`
    if possible and return result as a new dictionary. If a value is of the
    form `val:x`, then this value will be set to `x` and parsed
    appropriately.

    :param source: source dictionary
    :param mapping: mapping
    :return: mapping dictionary
    """
    result = {}
    for key, val in mapping.items():
        if len(val) > 5 and val[:4] == 'val:':
            val_parsed = val[4:]
            if val_parsed in val_map:
                result[key] = val_map[val_parsed]
            else:
                result[key] = val_parsed
        elif val in source and source[val] != 'undefined':
            result[key] = source[val]
    return result


class Plotter:
    """
    Sane and consistent configuration of plots.

    :param \*\*kw_args: configuration adjustments
    """
    _config = {'axes_color': 'black',
               'axes_width': 1,
               'axes_labelpad': 8,
               'grid_color': '0.8',
               'grid_width': .5,
               'grid_style': '--',
               'font_family': 'Adobe Caslon Pro',
               'font_size': 11,
               'fig_size': (8, 6),
               'legend_color': '0.95',
               'line_color': 'undefined',
               'marker_style': 'None',
               'line_width': 1,
               'line_style': '-',
               'colorbar_shrink': 0.8,
               'colorbar_aspect': 20,
               'marker_color': 'undefined',
               'marker_size': 5,
               'surface_rstride': 1,
               'surface_cstride': 1,
               'surface_antialiased': False,
               'surface_line_width': 0,
               'cmap': mpl.cm.coolwarm,
               'fill_alpha': 0.25,
               'fill_color': 'undefined',
               'label': 'undefined'
               }

    def __init__(self, **kw_args):
        self._generate_abbrevs()
        self._config.update(self._deabbrev(kw_args))
        self._config_global()
        self.plt = plt
        self._first = True
        self.plots = []

    def _abbrev(self, xs, pref=''):
        prefs = set([pref + x[len(pref)] for x in xs])
        prefixed = [(x, list(set(filter(lambda y: y.startswith(x), xs))))
                    for x in prefs]
        abbrevs = {}
        for pref, els in prefixed:
            if len(els) == 1 and len(pref) >= 2:
                abbrevs[pref] = els[0]
            else:
                abbrevs.update(self._abbrev(els, pref=pref))
        return abbrevs

    def _generate_abbrevs(self):
        self._abbrevs = {}
        depths = {}
        for key in self._config.keys():
            words = key.split('_')
            depth = len(words)
            if depth not in depths:
                depths[depth] = [[] for i in range(depth)]
            for i in range(depth):
                depths[depth][i].append(words[i])
        self._abbrevs = {}
        for depth, words in depths.items():
            pairs = [x.items() for x in map(self._abbrev, words)]
            for abbrevs in itertools.product(*pairs):
                glued = ['_'.join(x) for x in zip(*abbrevs)]
                self._abbrevs[glued[0]] = glued[1]

    def figure(self, *args, **kw_args):
        self._first = False
        self._config_figure(*args, **kw_args)
        self._config_specifics()
        return self

    def _config_global(self):
        mpl.rcParams.update({#'toolbar': 'None',
                             'figure.autolayout': True})
        mpl.rc('font', family=self._config['font_family'],
               size=self._config['font_size'])
        return self

    def _deabbrev(self, kw_args):
        return {self._abbrevs[k] if k in self._abbrevs else k: v
                for k, v in kw_args.items()}

    def _config_figure(self, *args, **kw_args):
        config = dict(self._config)
        config.update(self._deabbrev(kw_args))
        self.fig = plt.figure(*args,
                              facecolor='white',
                              edgecolor='white',
                              figsize=config['fig_size'])
        return self

    @abc.abstractmethod
    def _config_specifics(self):
        pass

    def show(self, *args, **kw_args):
        plt.show(*args, **kw_args)
        return self

    def save(self, fn, *args, **kw_args):
        # Save, and ignore warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.fig.savefig(fn, *args, **kw_args)
        return self

    def show_legend(self):
        self.leg = self.ax.legend(loc='upper right')
        self.leg.get_frame().set_color(self._config['legend_color'])
        return self

    def title(self, *args, **kw_args):
        plt.title(*args, **kw_args)
        return self

    def labels(self, x=None, y=None, z=None):
        if x is not None:
            self.ax.set_xlabel(x)
        if y is not None:
            self.ax.set_ylabel(y)
        if z is not None:
            self.ax.set_zlabel(z)
        return self

    def lims(self, x=None, y=None, z=None):
        if x is not None:
            self.ax.set_xlim(*x)
        if y is not None:
            self.ax.set_ylim(*y)
        if z is not None:
            self.ax.set_zlim(*z)
        return self

    def subplot(self, *args, **kw_args):
        self._check_first_figure()
        plt.subplot(*args, **kw_args)
        self._config_specifics()
        return self

    def subplot2grid(self, *args, **kw_args):
        self._check_first_figure()
        plt.subplot2grid(*args, **kw_args)
        self._config_specifics()
        return self

    def _check_first_figure(self):
        if self._first:
            self.figure()
            self._first = False

    def hide(self, x=False, y=False, z=False):
        if x:
            self.ax.xaxis.set_visible(False)
            self.ax.xaxis.set_ticklabels([])
        if y:
            self.ax.yaxis.set_visible(False)
            self.ax.yaxis.set_ticklabels([])
        if z:
            self.ax.zaxis.set_visible(False)
            self.ax.zaxis.set_ticklabels([])


class Plotter2D(Plotter):
    def __init__(self, **kw_args):
        Plotter.__init__(self, **kw_args)

    def _config_specifics(self):
        self._config_axes()
        self._config_ticks()
        return self

    def _config_axes(self):
        self.ax = plt.gca()
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color(self._config['axes_color'])
        self.ax.spines['bottom'].set_color(self._config['axes_color'])
        plt.grid(which='major',
                 linestyle=self._config['grid_style'],
                 color=self._config['grid_color'],
                 linewidth=self._config['grid_width'])
        self.ax.set_axisbelow(True)
        return self

    def _config_ticks(self):
        self.ax.xaxis.set_tick_params(width=1,
                                      color=self._config['axes_color'],
                                      right='off')
        self.ax.yaxis.set_tick_params(width=1,
                                      color=self._config['axes_color'],
                                      top='off')
        return self

    def plot(self, *args, **kw_args):
        self._check_first_figure()
        config = dict(self._config)
        config.update(self._deabbrev(kw_args))
        mapping = {'linewidth': 'line_width',
                   'linestyle': 'line_style',
                   'color': 'line_color',
                   'marker': 'marker_style',
                   'markerfacecolor': 'marker_color',
                   'markeredgecolor': 'marker_color',
                   'markersize': 'marker_size',
                   'label': 'label'}
        p = plt.plot(*args, **dict_map(config, mapping))
        self.plots.append(p)
        return self

    def fill(self, x, y1, y2, *args, **kw_args):
        self._check_first_figure()
        config = dict(self._config)
        config.update(self._deabbrev(kw_args))
        mapping = {'alpha': 'fill_alpha',
                   'edgecolor': 'val:none',
                   'facecolor': 'fill_color',
                   'interpolate': 'val:True'}
        p = plt.fill_between(x, y1, y2, *args, **dict_map(config, mapping))
        self.plots.append(p)
        return self


class Plotter3D(Plotter):
    def __init__(self, **kw_args):
        Plotter.__init__(self, **kw_args)

    def _config_specifics(self):
        self._config_axes()
        self._config_ticks()
        return self

    def _config_axes(self):
        self.ax = plt.gca(projection='3d')
        for ax in [self.ax.w_xaxis, self.ax.w_yaxis, self.ax.w_zaxis]:
            ax.set_pane_color(to_rbga('white'))
            ax.line.set_color(self._config['axes_color'])
            ax.line.set_lw(self._config['axes_width'])
            ax._axinfo['grid']['color'] = to_rbga(self._config['grid_color'])
            ax.gridlines.set_lw(self._config['grid_width'])
        return self

    def _config_ticks(self):
        for ax in [self.ax.xaxis, self.ax.yaxis, self.ax.zaxis]:
            ax.labelpad = self._config['axes_labelpad']
            ax._axinfo['tick']['inward_factor'] = 0
            ax._axinfo['tick']['outward_factor'] = 0.2
            ax.majorTicks[0].tick1line.set_color(self._config['axes_color'])
            ax.majorTicks[0].tick1line.set_linewidth(
                self._config['axes_width'])
        return self

    def show_colorbar(self, obj, **kw_args):
        config = {'shrink': self._config['colorbar_shrink'],
                  'aspect': self._config['colorbar_aspect']}
        config.update(self._deabbrev(kw_args))
        self.cb = self.fig.colorbar(obj, **config)
        self.cb.outline.set_linewidth(self._config['axes_width'])
        self.cb.ax.tick_params(right='off')
        return self

    def plot_surface(self, *args, **kw_args):
        self._check_first_figure()
        config = dict(self._config)
        config.update(self._deabbrev(kw_args))
        mapping = {'rstride': 'surface_rstride',
                   'cstride': 'surface_cstride',
                   'cmap': 'cmap',
                   'linewidth': 'surface_line_width',
                   'antialiased': 'surface_antialiased'}
        self.ax.plot_surface(*args, **dict_map(config, mapping))
        return self

    def view(self, elevation, azimuth):
        self.ax.view_init(elev=elevation, azim=azimuth)
        return self
