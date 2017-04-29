# weibull module for fitting Weibull psychometric function

# TODOs:
# Weibull class:
# [ ] - refactor for ease of use
#       -> think about mirroring sklearn API


import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from utils import trim, trim_df, round2step, reformat_params
from viz import plot_weibull


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

	def __init__(self, x, y, method='Nelder-Mead', kind='weibull', bounds=None):
		self.x = x
		self.orig_y = y.copy()
		# y is 0 or 1 - this is problematic for log
		# so we drag the values a little
		self.y = self.drag(y)

		# method (optimizer)
		valid_methods = ('Nelder-Mead', 'L-BFGS-B', 'TNC', 'SLSQP')
		if method in valid_methods:
			self.method = method
		else:
			raise ValueError('method must be one of {}, got {} '
							 'instead.'.format(valid_methods, method))
		# kind
		valid_kinds = ('weibull', 'generalized logistic')
		if kind in valid_kinds:
			self.kind = kind
		else:
			raise ValueError('kind must be one of {}, got {} '
							 'instead.'.format(valid_kinds, kind))

		min_float = sys.float_info.min
		if self.kind == 'weibull':
			self._fun = weibull
			self.bounds = ((min_float, None), (min_float, None),
						   (min_float, 0.5)) if bounds is None else bounds
		elif self.kind == 'generalized logistic':
			self._fun = generalized_logistic
			self.bounds = ((0.5, 1.), (0., None), (min_float, None),
						   (None, None), (None, None))

	def fun(self, params):
		return self._fun(self.x, params)

	def predict(self, X):
		return self._fun(X, self.params)

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

	def fit(self, initparams):
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
		return minimize(invfun, self.params[1], method='Nelder-Mead')['x'][0]

	def get_threshold(self, corr):
		return list(map(self._inverse, corr))

	def plot(self, pth='', ax=None, points=True, line=True, mean_points=False,
			 min_bucket='adaptive', split_bucket='adaptive', line_color=None,
			 contrast_steps=None, mean_points_color=(0.22, 0.58, 0.78)):
		return plot_weibull(self, pth=pth, ax=ax, points=points, line=line,
							mean_points=mean_points, min_bucket=min_bucket,
							split_bucket=split_bucket, line_color=line_color,
							contrast_steps=contrast_steps,
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
			b, t, lapse = params
		else:
			b, t = params
			lapse = 0.

		k = ( -np.log((1.0 - corr_at_thresh) / (1.0 - chance_level)) ) \
			** (1.0 / b)
		expo = ((k * x) / t) ** b

		return (1 - lapse) - (1 - lapse - chance_level) * np.exp(-expo)


def generalized_logistic(x, params, chance_level=0.5, C=1.):
	'''
	A - lower asymptote
	K - upper asymptote
	B - growth rate
	v - where maximum growth occurs (which asymptote)
	Q - Y(0)
	C - another scaling parameter
	'''
	A = chance_level
	K, B, v, Q = params
	return A + (K - A) / ((C + Q * np.exp(-B * x)) ** (1 / v))



class QuestPlus(object):

    def __init__(self, stim, params, function=weibull):
        self.function = function
        self.stim_domain = stim
        self.param_domain = reformat_params(params)

        self._param_orig_shape = (list(map(len, params)) if
                                  isinstance(params, list) else len(params))
        self._stim_orig_shape = (list(map(len, params)) if
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

    def update(self, contrast, ifcorrect):
        '''update posterior probability with outcome of current trial.

        contrast - contrast value for the given trial
        ifcorrect   - whether response was correct or not
                      1 - correct, 0 - incorrect
        '''

        # turn ifcorrect to response index
        resp_idx = 1 - ifcorrect
        contrast_idx = np.where(self.stim_domain == contrast)[0][0]

        # take likelihood of such resp for whole model parameter domain
        likelihood = self.likelihoods[contrast_idx, :, resp_idx]
        self.posterior *= likelihood
        self.posterior /= self.posterior.sum()

        # log history of contrasts and responses
        self.stim_history.append(contrast)
        self.resp_history.append(ifcorrect)

    def next_contrast(self):
        '''get contrast value that minimizes posterior entropy'''
        #
        # compute next trial contrast
        full_posterior = self.likelihoods * self.posterior[
            np.newaxis, :, np.newaxis]
        norm = full_posterior.sum(axis=1, keepdims=True)
        full_posterior /= norm

        H = -np.nansum(full_posterior * np.log(full_posterior), axis=1)
        self.entropy = (norm[:, 0, :] * H).sum(axis=1)

        # choose contrast by min arg
        return self.stim_domain[self.entropy.argmin()]


# for interactive plotting:
# -------------------------
# - [ ] clean these functions below - could be one function for all
def fitw(df, ind=None, last=60, init_params=[1., 1.], method='Nelder-Mead'):
	if ind is None and last:
		n_rows = df.shape[0]
		start_ind = min([1, n_rows - last + 1])
		ind = np.arange(start_ind, n_rows + 1, dtype = 'int')

	x = df.loc[ind, 'opacity'].values.astype('float64')
	y = df.loc[ind, 'ifcorrect'].values.astype('int32')

	# fit on non-nan
	notnan = ~(np.isnan(y))
	w = Weibull(x, y, method=method)
	w.fit(init_params)
	return w
