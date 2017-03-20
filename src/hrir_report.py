#!/usr/bin/env python
import pickle
import numpy as np

import core.out as out


def get(reports, keys):
    def getter(x):
        for k in keys:
            x = x[k]
        return x

    vals = [getter(x) for x in reports]
    return {'mean': np.mean(vals),
            'std': np.std(vals) / len(vals) ** .5}


def fetch(paths):
    reports = []
    for path in paths:
        with open(path) as f:
            reports.append(pickle.load(f))
    return reports


seeds = [103, 104, 105, 106, 107, 108, 109, 112, 113, 114]

base_path = 'output/stats/hrir/causal-model={},post=n,resample={}.pickle'
reports_cgpcm = fetch([base_path.format('y', i) for i in seeds])
reports_gpcm = fetch([base_path.format('n', i) for i in seeds])


def subreport(keys_mf, keys_smf):
    return {'GPCM': {'MF': get(reports_gpcm, keys_mf),
                     'SMF': get(reports_gpcm, keys_smf)},
            'CGPCM': {'MF': get(reports_cgpcm, keys_mf),
                      'SMF': get(reports_cgpcm, keys_smf)}}


def prediction_report(key):
    return {'SMSE': subreport(['prediction', key, 'SMSE', 'MF'],
                              ['prediction', key, 'SMSE', 'SMF']),
            'MLL': subreport(['prediction', key, 'MLL', 'MF'],
                             ['prediction', key, 'MLL', 'SMF'])}


report = {'ELBO': subreport(['parameters', 'ELBO MF'],
                            ['parameters', 'ELBO SMF']),
          'filter': prediction_report('filter'),
          'kernel': prediction_report('kernel'),
          'function': prediction_report('function')}

out.dict_(report)
