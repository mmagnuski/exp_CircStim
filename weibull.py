# weibull module for fitting Weibull psychometric function

# TODOs:
# Weibull class:
# [ ] - refactor for ease of use
#       -> think about mirroring sklearn API


from __future__ import absolute_import
import os
import sys
import random
from functools import partial
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from .utils import trim, trim_df, round2step, reformat_params
from .viz import plot_weibull, plot_quest_plus


# from dB and to dB utility functions:
to_db = lambda x: 10 * np.log10(x)
from_db = lambda x: 10 ** (x / 10.)

class Weibull:
    '''
    Weibull is a simple class for fitting and
    evaluating Weibull psychometric function.

    example:
    --------
    w = Weibull(stim_intensity, if_resp_correct)
    initparams = [1., 1.]
    w.fit(initparams)
    final_params = w.params
    '''

    def __init__(self, method='Nelder-Mead', kind='weibull', bounds=None,
                 corr_at_thresh=0.75):
        self.x = None
        self.y = None
        self.orig_y = None
        self.params = None
        self.corr_at_thresh = corr_at_thresh

        # method (optimizer)
        valid_methods = ('Nelder-Mead', 'L-BFGS-B', 'TNC', 'SLSQP')
        if method in valid_methods:
            self.method = method
        else:
            raise ValueError('method must be one of {}, got {} '
                             'instead.'.format(valid_methods, method))
        # kind
        valid_kinds = ('weibull', 'weibull_db', 'generalized logistic')
        if kind in valid_kinds:
            self.kind = kind
        else:
            raise ValueError('kind must be one of {}, got {} '
                             'instead.'.format(valid_kinds, kind))

        min_float = sys.float_info.min
        if self.kind == 'weibull':
            self._fun = weibull
            self.bounds = ((min_float, None), (min_float, None),
                           (min_float, 0.15)) if bounds is None else bounds
        elif self.kind == 'weibull_db':
            self._fun = weibull_db
            self.bounds = ((-40., None), (min_float, None),
                           (min_float, 0.15)) if bounds is None else bounds
        elif self.kind == 'generalized logistic':
            self._fun = generalized_logistic
            self.bounds = ((0.5, 1.), (0., None), (min_float, None),
                           (None, None), (None, None))

    def fun(self, params):
        return self.predict(self.x, params=params)

    def predict(self, X, params=None):
        if params is None:
            params = self.params

        if self.kind == 'weibull':
            return self._fun(X, params, corr_at_thresh=self.corr_at_thresh)
        else:
            return self._fun(X, params)

    def drag(self, y):
        return y * .99 + .005

    def loglik_ned(self, params):
        # if params are not within their bounds, return some high value
        n_params = len(params)
        bounds = self.bounds[:n_params]
        if any([outside_bounds(val, bound) for val, bound
                                           in zip(params, bounds)]):
            return 10000.

        # return negative log-likelihood
        return self.loglik(params)

    def loglik(self, params):
        y_pred = self.fun(params)
        return np.nansum(np.log(y_pred) * self.orig_y +
                      np.log(1 - y_pred) * (1 - self.orig_y)) * -1.

    def fit(self, x, y, initparams):
        self.x = x
        self.orig_y = y.copy()
        # y is 0 or 1 - this is problematic for log
        # so we drag the values a little
        self.y = self.drag(y)

        n_params = len(initparams)
        if self.method == 'Nelder-Mead':
            self.params = minimize(self.loglik_ned, initparams,
                                   method='Nelder-Mead')['x']
        else:
            # use bounds
            self.params = minimize(self.loglik, initparams, method=self.method,
                                   bounds=self.bounds[:n_params])['x']

    def _inverse(self, corrinput):
        invfun = lambda cntr: (corrinput - self.predict(cntr)) ** 2
        # optimize with respect to correctness
        start_param = self.params[0]
        return minimize(invfun, start_param, method='Nelder-Mead')['x'][0]

    def get_threshold(self, corr):
        return list(map(self._inverse, corr))

    def plot(self, x=None, pth='', ax=None, points=True, line=True,
             mean_points=False, min_bucket='adaptive', split_bucket='adaptive',
             line_color=None, contrast_steps=None, linewidth=3.,
             mean_points_color=(0.22, 0.58, 0.78)):
        return plot_weibull(
            self, x=x, pth=pth, ax=ax, points=points, line=line,
            mean_points=mean_points, min_bucket=min_bucket,
            split_bucket=split_bucket, line_color=line_color,
            contrast_steps=contrast_steps, linewidth=linewidth,
            mean_points_color=mean_points_color)


def outside_bounds(val, bounds):
    if bounds[0] is not None and val < bounds[0]:
        return True
    if bounds[1] is not None and val > bounds[1]:
        return True
    return False


def weibull(x, params, corr_at_thresh=0.75, chance_level=0.5):
        # unpack params
        if len(params) == 3:
            t, b, lapse = params
        else:
            t, b = params
            lapse = 0.

        k = ( -np.log((1.0 - corr_at_thresh) / (1.0 - chance_level)) ) \
            ** (1.0 / b)
        expo = ((k * x) / t) ** b

        return (1 - lapse) - (1 - lapse - chance_level) * np.exp(-expo)


def weibull_db(contrast, params, guess=0.5):
    # unpack params
    if len(params) == 3:
        threshold, slope, lapse = params
    else:
        threshold, slope = params
        lapse = 0.

    return (1 - lapse) - (1 - lapse - guess) * np.exp(
        -10. ** (slope * (contrast - threshold) / 20.))


# TODO:
# - [ ] highlight lowest point in entropy in plot
class QuestPlus(object):
    def __init__(self, stim, params, function=weibull):
        self.function = function
        self.stim_domain = stim
        self.param_domain = reformat_params(params)

        self._orig_params = deepcopy(params)
        self._orig_param_shape = (list(map(len, params)) if
                                  isinstance(params, list) else len(params))
        self._orig_stim_shape = (list(map(len, params)) if
                                 isinstance(params, list) else len(params))

        n_stim, n_param = self.stim_domain.shape[0], self.param_domain.shape[0]

        # setup likelihoods for all combinations
        # of stimulus and model parameter domains
        self.likelihoods = np.zeros((n_stim, n_param, 2))
        for p in range(n_param):
            self.likelihoods[:, p, 0] = self.function(self.stim_domain,
                                                      self.param_domain[p, :])

        # assumes (correct, incorrect) responses
        self.likelihoods[:, :, 1] = 1. - self.likelihoods[:, :, 0]

        # we also assume a flat prior (so we init posterior to flat too)
        self.posterior = np.ones(n_param)
        self.posterior /= self.posterior.sum()

        self.stim_history = list()
        self.resp_history = list()
        self.entropy = np.ones(n_stim)

    def update(self, contrast, ifcorrect, approximate=False):
        '''update posterior probability with outcome of current trial.

        contrast - contrast value for the given trial
        ifcorrect   - whether response was correct or not
                      1 - correct, 0 - incorrect
        '''

        # turn ifcorrect to response index
        resp_idx = 1 - ifcorrect
        contrast_idx = self._find_contrast_index(
            contrast,  approximate=approximate)[0]

        # take likelihood of such resp for whole model parameter domain
        likelihood = self.likelihoods[contrast_idx, :, resp_idx]
        self.posterior *= likelihood
        self.posterior /= self.posterior.sum()

        # log history of contrasts and responses
        self.stim_history.append(contrast)
        self.resp_history.append(ifcorrect)

    def _find_contrast_index(self, contrast, approximate=False):
        contrast = np.atleast_1d(contrast)
        if not approximate:
            idx = [np.nonzero(self.stim_domain == cntrst)[0][0]
                   for cntrst in contrast]
        else:
            idx = np.abs(self.stim_domain[np.newaxis, :] -
                         contrast[:, np.newaxis]).argmin(axis=1)
        return idx

    def next_contrast(self, axis=None):
        '''Get contrast value minimizing entropy of the posterior
        distribution.

        Expected entropy is updated in self.entropy.

        Returns
        -------
        contrast : contrast value for the next trial.'''
        full_posterior = self.likelihoods * self.posterior[
            np.newaxis, :, np.newaxis]
        if axis is not None:
            shp = full_posterior.shape
            new_shape = [shp[0]] + self._orig_param_shape + [shp[-1]]
            full_posterior = full_posterior.reshape(new_shape)
            reduce_axes = np.arange(len(self._orig_param_shape)) + 1
            reduce_axes = tuple(np.delete(reduce_axes, axis))
            full_posterior = full_posterior.sum(axis=reduce_axes)

        norm = full_posterior.sum(axis=1, keepdims=True)
        full_posterior /= norm

        H = -np.nansum(full_posterior * np.log(full_posterior), axis=1)
        self.entropy = (norm[:, 0, :] * H).sum(axis=1)

        # choose contrast with minimal entropy
        return self.stim_domain[self.entropy.argmin()]

    # TODO:
    # - [ ] check correctness
    def compute_entropy(self, contrasts):
        '''Compute entropy for a sequence of contrasts.'''

        contrast_idx = self._find_contrast_index(contrasts)
        agg_posterior = (np.ones(self.likelihoods[0].shape) *
                         self.posterior[:, np.newaxis])
        for idx in contrast_idx:
            agg_posterior *= self.likelihoods[idx]
        norm = agg_posterior.sum(axis=0, keepdims=True)
        agg_posterior /= norm

        H = -np.nansum(agg_posterior * np.log(agg_posterior), axis=0)
        entropy = (norm[0, :] * H).sum()
        return entropy

    def get_posterior(self):
    	return self.posterior.reshape(self._orig_param_shape)

    def get_fit_params(self, select='mode', weibull_args=None):
        if select in ['max', 'mode']:
            # parameters corresponding to maximum peak in posterior probability
            return self.param_domain[self.posterior.argmax(), :]
        elif select == 'mean':
            # parameters weighted by their probability
            return (self.posterior[:, np.newaxis] *
                    self.param_domain).sum(axis=0)
        elif select == 'ML':
            if weibull_args is None:
                weibull_args = dict(kind='weibull_db')
            w = Weibull(**weibull_args)
            init_params = self.get_fit_params(select='max')
            w.fit(np.array(self.stim_history), np.array(self.resp_history),
                           init_params)
            return w.params

    def fit(self, contrasts, responses, approximate=False):
        for contrast, response in zip(contrasts, responses):
            self.update(contrast, response, approximate=approximate)

    def plot(self):
        '''Plot posterior model parameter probabilities and weibull fits.'''
        return plot_quest_plus(self)


def init_thresh_optim(df, qp, model_params, logger=None):
    '''Initialize threshold optimization.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing behavioral data. Must contain opacity and
        ifcorrect columns.
    qp : QuestPlus instance
        QuestPlus fitted to the behavioral data.

    Returns
    -------
    qps : list of QuestPlus instances
        List of QuestPlus objects, each tracking a different correctness
        threshold.
    corrs : array of float
        Correctness thresholds for consecutive QuestPlus objects.
    '''
    init_params = qp.get_fit_params()
    weib = Weibull(kind='weibull')
    weib.fit(df.loc[:, 'opacity'], df.loc[:, 'ifcorrect'], init_params)
    lapse = weib.params[-1]
    top_corr = max(0.9, 1 - lapse - 0.01)
    if logger: logger.write('top correctness: {}'.format(top_corr))
    low, hi = weib.get_threshold([0.51, top_corr])
    low, hi = [max(low, 0.001), min(2., hi)]
    if logger:
        msg = 'low (51%) and high (top corr) thresholds: {}'
        logger.write(msg.format(low, hi))
    rng = (hi - low)
    widen = min(0.08, rng * 0.1)

    model_thresholds, model_slopes, model_lapses = model_params
    stim_params = np.linspace(max(0.001, low - widen),
                              min(hi + widen, 1.5), num=120)
    if logger:
        msg = 'stim params for all qps: {}'
        msg.format(stim_params)

    # fit QuestPlus for each threshold (takes ~ 6 - 11 seconds)
    qps = list()
    corrs = np.linspace(0.6, min(0.9, top_corr), num=5)
    param_space = [model_thresholds, model_slopes, model_lapses]
    for corr in corrs:
        this_wb = partial(weibull, corr_at_thresh=corr)
        qp = QuestPlus(stim_params, param_space, function=this_wb)
        qp.fit(df.loc[:, 'opacity'], df.loc[:, 'ifcorrect'], approximate=True);
        qps.append(qp)

    return corrs, qps


def plot_threshold_entropy(qps, corrs=None, axis=None):
    '''Plot entropy for each threshold.'''
    if corrs is None:
        corrs = ['step {}'.format(idx) for idx in range(1, len(qps) + 1)]
    if axis is None:
        axis = plt.gca()

    posteriors = [qp.get_posterior().sum(axis=(1, 2)) for qp in qps]
    for post, corr in zip(posteriors, corrs):
        lines = axis.plot(qps[-1]._orig_params[0], post, label=str(corr))
        color = lines[0].get_color()
        max_idx = post.argmax()
        axis.scatter(qps[-1]._orig_params[0][max_idx], post[max_idx],
                     facecolor=color, edgecolor='k', zorder=10, s=50)

    axis.legend()
    return axis


class PsychometricMonkey(object):
    def __init__(self, psychometric=None, response_mapping=None,
                 intensity_var=None, stimulus_var=None):
        if psychometric is None:
            # init some reasonable psychometric function
            psychometric =  Weibull(kind='weibull')
            psychometric.params = [0.1, 4.5, 0.05]

        self.psychometric = psychometric
        self.response_mapping = response_mapping
        self.stimulus_var = stimulus_var
        self.intensity_var = intensity_var
        self.response_keys = list(set(response_mapping.values()))

    def respond(self, trial):
        prob_correct = self.psychometric.predict(trial[self.intensity_var])
        random_val = np.random.rand()
        correct_response = self.response_mapping[trial[self.stimulus_var]]
        if random_val <= prob_correct:
            return correct_response
        else:
            incorrect_keys = list(self.response_keys)
            incorrect_keys.remove(correct_response)
            return random.sample(incorrect_keys, 1)[0]
