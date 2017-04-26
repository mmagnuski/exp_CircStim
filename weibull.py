# weibull module for fitting Weibull psychometric function

# TODOs:
# Weibull class:
# [ ] - refactor for ease of use
#       -> do not have to pass additional args
#          if they are in the model
#       -> but allow to pass these args if needed
#       -> think about mirroring sklearn API
# [ ] - predict (now performed by _fun but could change names)
# [ ] - ! think about adding params to predictors for both slope and position


import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from utils import trim, trim_df, round2step, group, check_color, reformat_params


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
		self.orig_y = y
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
		return np.sum(np.log(y_pred) * self.orig_y +
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
		# get predicted data
		numpnts = 1000
		x = np.linspace(0., 2., num=numpnts)
		y = self.predict(x)

		# add noise to y data to increase visibility
		l = len(self.x)
		yrnd = np.random.uniform(-0.065, 0.065, l)

		if line_color is None:
			line_color = 'seaborn_red'

		# plot setup
		if ax is None:
			f, ax = plt.subplots()
		ax.set_axis_bgcolor((0.92, 0.92, 0.92))
		plt.hold(True) # just in case (matlab habit)
		plt.grid(True, color=(1., 1., 1.), lw=1.5, linestyle='-', zorder=-1)

		line_color = check_color(line_color)

		# plot buckets
		if mean_points:
			from scipy import stats
			mean_points_color = check_color(mean_points_color)

			# bucketize
			# ---------
			# get points and sort
			x_pnts = self.x.copy()
			y_pnts = self.orig_y.copy()
			srt = x_pnts.argsort()
			x_pnts = x_pnts[srt]
			y_pnts = y_pnts[srt]

			# adaptive min_bucket and split_bucket
			adaptive_min_bucket = isinstance(min_bucket, str) and \
				min_bucket == 'adaptive'
			adaptive_split_bucket = isinstance(split_bucket, str) and \
				split_bucket == 'adaptive'
			if adaptive_min_bucket or adaptive_split_bucket:
				drop_elements = int(np.ceil(len(x_pnts) * 0.08))
				contrast_range = x_pnts[drop_elements:-drop_elements][[0, -1]]
				contrast_range = np.diff(contrast_range)[0]
				if adaptive_min_bucket:
					min_bucket = contrast_range / 20.
				if adaptive_split_bucket:
					split_bucket = contrast_range / 10.

			# look for buckets
			x_buckets = group(np.diff(x_pnts) <= min_bucket)
			n_pnts_in_bucket = (np.diff(x_buckets, axis=-1) + 1).ravel()
			good_buckets = n_pnts_in_bucket >= (3 - 1) # -1 because of diff

			if x_buckets.shape[0] > 0 and np.any(good_buckets):
				x_buckets = x_buckets[good_buckets, :]

				# turn buckets to slices, get mean and sem
				x_buckets[:, 1] += 2 # +1 because of python slicing, another +1 because of diff
				slices = [slice(l[0], l[1]) for l in x_buckets]

				# test each slice for contrast range and split if needed
				add_slices = list()
				ii = 0

				while ii < len(slices):
					slc = slices[ii]
					pnts = x_pnts[slc]
					l, h = pnts[[0, -1]]
					rng = h - l
					start = slc.start
					if rng > split_bucket:
						slices.pop(ii)
						n_full_splits = int(np.floor(rng / split_bucket))
						last_l = l
						last_ind = 0
						for splt in range(n_full_splits):
							this_high = last_l + split_bucket
							this_ind = np.where(pnts >= this_high)[0][0]
							add_slices.append(slice(start + last_ind, start + this_ind + 1))
							last_l = this_high
							last_ind = this_ind + 1

						# last one - to the end
						if start + last_ind < slc.stop:
							add_slices.append(slice(start + last_ind, slc.stop))
					else:
						ii += 1
				slices.extend(add_slices)

				x_bucket_mean = np.array([x_pnts[slc].mean()
										  for slc in slices])
				y_bucket_mean = np.array([y_pnts[slc].mean()
										  for slc in slices])
				bucket_sem = np.array([stats.sem(y_pnts[slc])
									   for slc in slices])

				# plot bucket means and sem
				plt.scatter(x_bucket_mean, y_bucket_mean, lw=0, zorder=4, s=32.,
							c=mean_points_color)
				plt.vlines(x_bucket_mean,
						   y_bucket_mean - bucket_sem,
						   y_bucket_mean + bucket_sem,
						   lw=2, zorder=4, colors=mean_points_color)

		if contrast_steps is not None:
			corrs = self.predict(contrast_steps)
			plt.vlines(contrast_steps, corrs - 0.04, corrs + 0.04,
					   lw=2, zorder=4, colors=[0., 0., 0.])

		if points:
			plt.scatter(self.x, self.orig_y + yrnd, alpha=0.6, lw=0,
				zorder=6, c=[0.3, 0.3, 0.3])
		if line:
			plt.plot(x, y, zorder=5, lw=3, color=line_color)

		# aesthetics
		# ----------
		gab99 = self.get_threshold([0.95])[0]
		if gab99 < 0. or gab99 > 2.:
			maxval = 2.
		else:
			maxval = gab99 + 0.1
		if self.params[0] <= 0:
			maxval = np.max(self.x) + 0.1
		uplim = min([2.0, np.round(maxval, decimals=1)])
		plt.xlim([0.0, uplim])
		plt.ylim([-0.1, 1.1])
		plt.xlabel('stimulus intensity')
		plt.ylabel('correctness')

		# save figure
		if pth:
			tempfname = os.path.join(pth, 'weibull_fit_temp.png')
			plt.savefig(tempfname, dpi=120)
			plt.close()
			return tempfname


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


def fit_weibull(db, i):
	take_last = min([i - 15, 60])
	idx = np.array(np.arange(i - take_last + 1, i + 1), dtype = 'int')
	ifcorr = db.loc[idx, 'ifcorrect'].values.astype('int')
	opacit = db.loc[idx, 'opacity'].values.astype('float')

	# fit on non-nan
	notnan = ~(np.isnan(ifcorr))
	w.fit([1.0, 1.0])
	w = Weibull(opacit[notnan], ifcorr[notnan])
	return w



class QuestPlus(object):

    def __init__(self, stim, params, function=weibull):
        self.function = function
        self.stim_domain = stim
        self.param_domain = reformat_params(params)

        n_stim, n_param = self.stim_domain.shape[0], self.param_domain.shape[0]

        # setup likelihoods for all combinations
        # of stimulus and model parameter domains
        likelihoods = np.zeros((c.shape[0], params.shape[0], 2))
        for p in range(params.shape[0]):
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

        return self

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


# TODO:
# - check if model is necessary, if not - simplify
# - maybe add 'random' to steps method this would shift the contrast points
#   randomly from -0.5 to 0.5 of respective bin width (asymmteric in logsteps)
def get_new_contrast(model, vmin=0.01, corr_lims=[0.55, 0.95],
					 contrast_lims=None, method='4steps', discrete_steps=False):
	'''
	method : string
		* '5steps', '6steps', '12steps' - divides the contrast
		  range to equally spaced steps. The number of the
		  steps is defined at the beginning of the string.
		* '6logsteps' - six logaritmic steps
		* '4midlogsteps' - four log steps from the middle point
		  of the contrast range in each direction (left and
		  right)

	* if contrast_lims are not set and model.params[0] <= 0 then contrast is chosen
	from range [vmin, vmin + 0.2]
	* if model has lapse rate and corr_lims[1] > (1 - lapse_rate) then corr_lims[1]
	is set to (1 - lapse_rate) - 0.01
	'''

	assert 'steps' in method

	# get method details from string
	log = 'log' in method
	steps = ''
	for c in method:
		if c.isdigit():
			steps += c
		else:
			break
	steps = int(steps) if steps else 5
		

	# correct high corr limit if it goes beyond 1 - lapse_rate
	if model is not None:
		if len(model.params) == 3:
			max_corr = 1 - model.params[2]
			if corr_lims is not None and corr_lims[1] > max_corr:
				corr_lims = list(corr_lims)
				corr_lims[1] = max_corr - 0.01

	# take contrast for specified correctness levels
	if contrast_lims is None:
		if model.params[0] <= 0: # should be some small positive value
			contrast_lims = [vmin, vmin + 0.2]
		else:
			contrast_lims = model.get_threshold(corr_lims)

	# get N values from the contrast range
	if log:
		if 'mid' in method:
			# add midpoint
			pnts = [contrast_lims[0], np.mean(contrast_lims),
				contrast_lims[1]]
			pw = np.log10(pnts)
			lft = np.logspace(pw[1], pw[0], num=steps+1)
			rgt = np.logspace(pw[1], pw[2], num=steps+1)
			check_contrast = np.hstack([lft[len(lft)::-1], rgt[1:]])
		else:
			pw = np.log10(contrast_lims)
			check_contrast = np.logspace(*pw, num=steps)
	else:
		check_contrast = np.linspace(*contrast_lims, num=steps)

	# trim all points
	check_contrast = np.array([trim(c, vmin, 1.)
		for c in check_contrast])

	# try different contrast steps for best granularity
	if discrete_steps is not False:
		steps = discrete_steps if isinstance(discrete_steps, list) else [discrete_steps]
		base_unique = len(check_contrast) == len(np.unique(check_contrast))
		if not base_unique:
			orig_check_contrast = check_contrast
			check_contrast, inverse = np.unique(check_contrast, return_inverse=True)
		for s in steps:
			this_contrast = round2step(check_contrast, step=s)
			this_contrast = np.array([trim(x, vmin, 1.)
				for x in this_contrast])
			new_nonrep = len(this_contrast) == len(np.unique(this_contrast))
			if new_nonrep:
				break
		if not base_unique:
			this_contrast = this_contrast[inverse]
	else:
		this_contrast = check_contrast

	return this_contrast, contrast_lims


def cut_df_corr(df, num_bins=7):
	# trim the dataframe
	df = trim_df(df)

	# make sure the type is int
	df.ifcorrect = df.ifcorrect.astype('int')

	# find bin with corr below:
	bins = pd.cut(df.opacity, num_bins)
	binval = df.groupby(bins)['ifcorrect'].mean()

	# get bin low and high:
	lowfun = lambda x: float(x.split(',')[0][1:])
	highfun = lambda x: float(x.split(',')[1][1:-1])
	low = list(map(lowfun, binval.index))
	high = list(map(highfun, binval.index))
	return binval, np.vstack([low, high]).T


def get_new_range(binval, bins, high_corr=0.8, low_corr=0.6):
	rng = np.where(binval > high_corr)[0]
	if np.any(rng):
		highval = bins[rng[0], 0]
	else:
		highval = bins[-1, 0]
	where_low = np.where(binval < low_corr)[0]
	if np.any(where_low):
		low = where_low[-1]
		lowval = bins[low, 0]
	else:
		go_below = bins[0, 1] - bins[0, 0]
		lowval = max(bins[0, 0] - go_below * 2, 0.03)
	return [lowval, highval]


def correct_weibull(model, num_fail, df=None):
	if not isinstance(df, pd.DataFrame):
		return None, num_fail
	if model.params[0] <= 0:
		num_fail += 1
		corr_below = max([1. - num_fail*0.1, 0.65])

		binval, bins = cut_df_corr(df, num_bins=7)
		contrast_range = get_new_range(binval, bins,
			high_corr=corr_below)
		return contrast_range, num_fail
	else:
		num_fail = 0
		contrast_range = None
		contrast_lims = model.get_threshold([0.55, 0.9])
		if contrast_lims[1] > 2.:
			# 0.9 is off the contrast limits
			binval, bins = cut_df_corr(df, num_bins=7)
			contrast_range = get_new_range(binval, bins)
		if contrast_lims[1] - contrast_lims[0] > 0.85:
			# the curve is very non-s-like, probably needs correction
			corr75 = model.get_threshold([0.75])
			if (contrast_lims[1] - corr75) > (corr75 - contrast_lims[0]):
				binval, bins = cut_df_corr(df, num_bins=7)
				contrast_range = get_new_range(binval, bins)

		return contrast_range, num_fail


# for interactive plotting:
# -------------------------
def fitw(df, ind, init_params=[1., 1.], method='Nelder-Mead'):
    x = df.loc[ind, 'opacity'].values.astype('float64')
    y = df.loc[ind, 'ifcorrect'].values.astype('int32')
    w = Weibull(x, y, method=method)
    w.fit(init_params)
    return w


def idx_at(fit_num):
    current_trial = 45 + (fit_num - 1) * 10
    take_last = min([current_trial - 15, 60])
    idx = np.array(np.arange(current_trial - take_last + 1,
                             current_trial + 1), dtype = 'int')
    return idx


def wfit_at(df, fit_num):
    idx = idx_at(fit_num)
    return fit_weibull(df, idx[-1])
