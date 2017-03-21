#!/usr/bin/env python
import pickle
import numpy as np
import scipy.stats as ss
import argparse

import core.out as out


def get(reports, keys):
    """
    Access each report in a list of reports according to a sequence of keys.

    :param reports: list of reports
    :param keys: sequence of keys
    :return: result of each report
    """

    def getter(x):
        for k in keys:
            x = x[k]
        return x

    vals = [getter(x) for x in reports]

    if type(vals[0]) == dict and 'estimate' in vals[0]:
        return [x['estimate'] for x in vals]
    else:
        return vals


def t_test(vals1, vals2, loss, name1, name2):
    """
    Check whether the mean of a variable is significantly better than that of
    another variable. Assumes that the samples are paired.

    :param vals1: samples of first variable
    :param vals2: samples of second variable
    :param loss: lower is better
    :param name1: name of first variable
    :param name2: name of second variable
    :return: second variable better
    """
    n = len(vals1)
    diffs = [x - y for x, y in zip(vals1, vals2)]
    std = np.std(diffs, ddof=1)
    mean = np.mean(diffs)
    t = mean / (std / n ** .5)  # Follows a t distribution with df = n - 1
    p = ss.t.cdf(t, n - 1)

    def message(p):
        return '{:3s} (p = {:.2e})'.format('yes' if p < 0.05 else 'no', p)

    return {'{} > {}'.format(name1, name2): message(p if loss else 1 - p),
            '{} < {}'.format(name1, name2): message(1 - p if loss else p)}


def fetch(paths):
    """
    Fetch report from a list of paths.

    :param paths: list of paths
    :return: reports
    """
    reports = []
    for path in paths:
        with open(path) as f:
            reports.append(pickle.load(f))
    return reports


def report_mf_smf(keys_mf, keys_smf, loss=True):
    """
    Generate a report that compares a MF result to a SMF result.

    :param keys_mf: keys sequence for the MF result
    :param keys_smf: keys sequence for the SMF result
    :param loss: lower is better
    :return: report
    """
    gpcm_mf = get(reports_gpcm, keys_mf)
    gpcm_smf = get(reports_gpcm, keys_smf)
    cgpcm_mf = get(reports_cgpcm, keys_mf)
    cgpcm_smf = get(reports_cgpcm, keys_smf)
    return {'GPCM': {'MF': np.mean(gpcm_mf),
                     'SMF': np.mean(gpcm_smf),
                     'significance': t_test(gpcm_mf,
                                            gpcm_smf,
                                            name1='MF',
                                            name2='SMF',
                                            loss=loss)},
            'CGPCM': {'MF': np.mean(cgpcm_mf),
                      'SMF': np.mean(cgpcm_smf),
                      'significance': t_test(cgpcm_mf,
                                             cgpcm_smf,
                                             name1='MF',
                                             name2='SMF',
                                             loss=loss)},
            'significance': {'MF': t_test(gpcm_mf,
                                          cgpcm_mf,
                                          name1='GPCM',
                                          name2='CGPCM',
                                          loss=loss),
                             'SMF': t_test(gpcm_smf,
                                           cgpcm_smf,
                                           name1='GPCM',
                                           name2='CGPCM',
                                           loss=loss)}}


def report_prediction(key):
    """
    Generate a report for a prediction.

    :param key: prediction key
    :return: report
    """
    return {'SMSE': report_mf_smf(['prediction', key, 'SMSE', 'MF'],
                                  ['prediction', key, 'SMSE', 'SMF']),
            'MLL': report_mf_smf(['prediction', key, 'MLL', 'MF'],
                                 ['prediction', key, 'MLL', 'SMF'])}


data_sets = {'hrir100': {'seeds': [102, 103, 104, 105, 106, 107, 108, 109, 112,
                                   113, 114],
                         'base_path': 'output/stats/hrir/causal-model={},'
                                      'post=n,resample={}.pickle'},
             'hrir200': {'seeds': [200, 201, 202, 203, 204, 206],
                         'base_path': 'output/stats/hrir/causal-model={},'
                                      'resample={}.pickle'}}

parser = argparse.ArgumentParser(description='Generate report.')
parser.add_argument('data_set', choices=data_sets.keys())
args = parser.parse_args()

seeds = data_sets[args.data_set]['seeds']
base_path = data_sets[args.data_set]['base_path']
reports_cgpcm = fetch([base_path.format('y', i) for i in seeds])
reports_gpcm = fetch([base_path.format('n', i) for i in seeds])
out.dict_({'ELBO': report_mf_smf(['parameters', 'ELBO MF'],
                                 ['parameters', 'ELBO SMF'],
                                 loss=False),
           'filter': report_prediction('filter'),
           'kernel': report_prediction('kernel'),
           'function': report_prediction('function')}, numeric_mod='.3e')
