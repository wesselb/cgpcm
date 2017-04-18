from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import abc
import warnings
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import matplotlib.pyplot as plt

import util
import data


def to_rbga(color, alpha=1.0):
    """
    Convert `color` to a tuple `(red, blue, green, alpha)`.

    :param color: color to convert
    :param alpha: alpha
    :return: converter color
    """
    return mpl.colors.colorConverter.to_rgba(color, alpha=alpha)


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


class Plotter(object):
    """
    Sane and consistent configuration of plots.

    :param \*\*kw_args: configuration adjustments
    """
    _config = {'axes_colour': 'black',
               'axes_width': 1,
               'axes_labelpad': 8,
               'grid_colour': '0.8',
               'grid_width': .5,
               'grid_style': '--',
               'font_family': 'Adobe Caslon Pro',
               'font_size': 11,
               'legend_colour': '0.95',
               'legend_alpha': 0.6,
               'legend_location': 'upper right',
               'line_colour': 'undefined',
               'line_width': 1,
               'line_style': '-',
               'colourbar_shrink': 0.8,
               'colourbar_aspect': 20,
               'marker_style': 'None',
               'marker_colour': 'undefined',
               'marker_face_colour': 'none',
               'marker_size': 5,
               'surface_rstride': 1,
               'surface_cstride': 1,
               'surface_antialiased': False,
               'surface_line_width': 0,
               'cmap': mpl.cm.coolwarm,
               'fill_alpha': 0.3,
               'fill_colour': 'undefined',
               'label': 'undefined',
               'figure_size': (8, 6),
               'figure_toolbar': 'toolbar2',  # None | toolbar2
               'figure_autolayout': True}

    def __init__(self, **kw_args):
        self._config.update(kw_args)
        self._config_global()
        self.plt = plt
        self._first = True
        self.plots = []
        self.x_scale = 1
        self.y_scale = 1
        self.z_scale = 1
        self.x_shift = 0
        self.y_shift = 0
        self.z_shift = 0

    def transform_x(self, x):
        return (x + self.x_shift) * self.x_scale

    def transform_y(self, y):
        return (y + self.y_shift) * self.y_scale

    def transform_z(self, z):
        return (z + self.z_shift) * self.z_scale

    def config(self, **kw_args):
        self._config.update(kw_args)

    def _map(self, mapping, kw_args):
        config = dict(self._config)
        config.update(kw_args)
        return dict_map(config, mapping)

    def figure(self, *args, **kw_args):
        self._first = False
        self._config_figure(*args, **kw_args)
        self._config_specifics()
        return self

    def _config_global(self):
        mapping = {'toolbar': 'figure_toolbar',
                   'figure.autolayout': 'figure_autolayout'}
        mpl.rcParams.update(**self._map(mapping, {}))
        mpl.rc('font',
               family=self._config['font_family'],
               size=self._config['font_size'])
        mpl.rc('text', usetex=True)
        return self

    def _config_figure(self, *args, **kw_args):
        mapping = {'facecolor': 'val:white',
                   'edgecolor': 'val:white',
                   'figsize': 'figure_size'}
        self.fig = plt.figure(*args, **self._map(mapping, kw_args))
        return self

    @abc.abstractmethod
    def _config_specifics(self):
        pass

    def show(self, *args, **kw_args):
        plt.show(*args, **kw_args)
        return self

    def save(self, fn, *args, **kw_args):
        util.mkdirs(fn)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.fig.savefig(fn, *args, **kw_args)
        return self

    def show_legend(self, **kw_args):
        config = self._map({'legend_colour': 'legend_colour',
                            'legend_alpha': 'legend_alpha',
                            'legend_location': 'legend_location'}, kw_args)
        self.leg = self.ax.legend(loc=config['legend_location'])
        self.leg.get_frame().set_color(config['legend_colour'])
        self.leg.get_frame().set_alpha(config['legend_alpha'])
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
            self.ax.set_xlim(*map(self.transform_x, x))
        if y is not None:
            self.ax.set_ylim(*map(self.transform_y, y))
        if z is not None:
            self.ax.set_zlim(*map(self.transform_z, z))
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

    def hide_ticks(self, x=False, y=False, z=False):
        if x:
            self.ax.xaxis.set_ticks([])
            self.ax.xaxis.set_ticklabels([])
        if y:
            self.ax.yaxis.set_ticks([])
            self.ax.yaxis.set_ticklabels([])
        if z:
            self.ax.zaxis.set_ticks([])
            self.ax.zaxis.set_ticklabels([])

    def ticks_arange(self, x=None, y=None, z=None):
        if x is not None:
            self.ax.xaxis.set_ticks(np.arange(*map(self.transform_x, x)))
        if y is not None:
            self.ax.yaxis.set_ticks(np.arange(*map(self.transform_y, y)))
        if z is not None:
            self.ax.zaxis.set_ticks(np.arange(*map(self.transform_z, z)))


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
        self.ax.spines['left'].set_color(self._config['axes_colour'])
        self.ax.spines['bottom'].set_color(self._config['axes_colour'])
        plt.grid(which='major',
                 linestyle=self._config['grid_style'],
                 color=self._config['grid_colour'],
                 linewidth=self._config['grid_width'])
        self.ax.set_axisbelow(True)
        return self

    def _config_ticks(self):
        self.ax.xaxis.set_tick_params(width=1,
                                      color=self._config['axes_colour'],
                                      right='off')
        self.ax.yaxis.set_tick_params(width=1,
                                      color=self._config['axes_colour'],
                                      top='off')
        return self

    def plot(self, x, y=None, *args, **kw_args):
        self._check_first_figure()
        mapping = {'linewidth': 'line_width',
                   'linestyle': 'line_style',
                   'color': 'line_colour',
                   'marker': 'marker_style',
                   'markerfacecolor': 'marker_face_colour',
                   'markeredgecolor': 'marker_colour',
                   'markersize': 'marker_size',
                   'label': 'label'}

        # Unpack if instance of `core.Data` is provided
        if y is None and type(x) == data.Data:
            x, y = x.x, x.y

        if y is None:
            p = plt.plot(self.transform_y(x),
                         *args, **self._map(mapping, kw_args))
        else:
            p = plt.plot(self.transform_x(x),
                         self.transform_y(y),
                         *args, **self._map(mapping, kw_args))
        self.plots.append(p)
        return self

    def scatter(self, x, y, *args, **kw_args):
        self._check_first_figure()
        mapping = {'marker': 'marker_style',
                   'c': 'marker_colour',
                   'edgecolors': 'marker_colour',
                   's': 'marker_size',
                   'label': 'label'}
        p = plt.scatter(self.x_scale * x,
                        self.y_scale * y,
                        *args, **self._map(mapping, kw_args))
        self.plots.append(p)
        return self

    def fill(self, x, y1, y2, *args, **kw_args):
        # Check which bound is upper bound
        if np.all(y1 >= y2):
            upper = y1
            lower = y2
        elif np.all(y2 >= y1):
            upper = y2
            lower = y1
        else:
            raise ValueError('either y1 >= y2 or y2 >= y1')

        self._check_first_figure()
        mapping = {'alpha': 'fill_alpha',
                   'edgecolor': 'val:none',
                   'facecolor': 'fill_colour',
                   'interpolate': 'val:True'}
        p = plt.fill_between(self.transform_x(x),
                             self.transform_y(lower),
                             self.transform_y(upper),
                             *args, **self._map(mapping, kw_args))
        self.plots.append(p)
        return self

    def tight(self):
        self.plt.axis('tight')
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
            ax.line.set_color(self._config['axes_colour'])
            ax.line.set_lw(self._config['axes_width'])
            ax._axinfo['grid']['color'] = to_rbga(self._config['grid_colour'])
            ax.gridlines.set_lw(self._config['grid_width'])
        return self

    def _config_ticks(self):
        for ax in [self.ax.xaxis, self.ax.yaxis, self.ax.zaxis]:
            ax.labelpad = self._config['axes_labelpad']
            ax._axinfo['tick']['inward_factor'] = 0
            ax._axinfo['tick']['outward_factor'] = 0.2
            ax.majorTicks[0].tick1line.set_color(self._config['axes_colour'])
            ax.majorTicks[0].tick1line.set_linewidth(
                self._config['axes_width'])
        return self

    def show_colourbar(self, obj, **kw_args):
        mapping = {'shrink': self._config['colourbar_shrink'],
                   'aspect': self._config['colourbar_aspect']}
        self.cb = self.fig.colorbar(obj, **self._map(mapping, kw_args))
        self.cb.outline.set_linewidth(self._config['axes_width'])
        self.cb.ax.tick_params(right='off')
        return self

    def plot_surface(self, x, y, z, *args, **kw_args):
        self._check_first_figure()
        mapping = {'rstride': 'surface_rstride',
                   'cstride': 'surface_cstride',
                   'cmap': 'cmap',
                   'linewidth': 'surface_line_width',
                   'antialiased': 'surface_antialiased'}
        self.ax.plot_surface(self.transform_x(x),
                             self.transform_y(y),
                             self.transform_z(z),
                             *args, **self._map(mapping, kw_args))
        return self

    def view(self, elevation, azimuth):
        self.ax.view_init(elev=elevation, azim=azimuth)
        return self
