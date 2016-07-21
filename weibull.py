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

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from utils import trim, trim_df, round2step


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

	def __init__(self, x, y):
		self.x = x
		self.orig_y = y
		# y is 0 or 1 - this is problematic for log
		# so we drag the values a little
		self.y = self.drag(y)

	def _fun(self, params, x, corr_at_thresh = 0.75, chance_level = 0.5):
		# unpack params
		b, t = params
		
		k = ( -np.log((1.0 - corr_at_thresh)/(1.0 - chance_level)) ) \
			** (1.0/b)

		expo = ((k * x)/t) ** b
		y = 1 - (1 - chance_level) * np.exp(-expo)
		return y
	
	def fun(self, params):
		return self._fun(params, self.x)

	def drag(self, y):
		return y * .99 + .005

	def loglik(self, params):
		y_pred = self.fun(params)
		# return negative log-likelihood
		return np.sum( np.log(y_pred)*self.orig_y + np.log(1-y_pred)*(1-self.orig_y) ) * -1.

	def fit(self, initparams):
		self.params = minimize(self.loglik, initparams, method='Nelder-Mead')['x']

	def _inverse(self, corrinput):
		invfun = lambda cntr: (corrinput - self._fun(self.params, cntr))**2
		# optimize with respect to correctness
		return minimize(invfun, self.params[1], method='Nelder-Mead')['x'][0]
	
	def get_threshold(self, corr):
		return map(self._inverse, corr)
	
	def plot(self, pth='', points=True, line=True):
		# get predicted data
		numpnts = 1000
		x = np.linspace(0., 2., num = numpnts)
		y = self._fun(self.params, x)

		# add noise to data to increase visibility
		l = len(self.x)
		yrnd = np.random.uniform(-0.05, 0.05, l)

		# plot setup
		f, ax = plt.subplots()
		ax.set_axis_bgcolor((0.92, 0.92, 0.92))
		plt.hold(True) # just in case (matlab habit)
		plt.grid(True, color=(1.,1.,1.), lw=1.5, linestyle='-', zorder = -1)

		# plot line
		if points:
			plt.scatter(self.x, self.orig_y + yrnd, alpha=0.6, lw=0, 
				zorder=4, c=[0.3, 0.3, 0.3])
		if line:
			plt.plot(x, y, zorder = 4, lw=3, color='k')

		# aesthetics
		gab99 = self.get_threshold([0.99])[0]
		if gab99 < 0. or gab99 > 2.:
			maxval = 2.
		else:
			maxval = gab99 + 0.1
		if self.params[0] <= 0:
			maxval = np.max(self.x) + 0.1
		uplim = min([2.0, np.round(maxval, decimals = 1)])
		plt.xlim([0.0, uplim])
		plt.ylim([-0.1, 1.1])
		plt.xlabel('stimulus intensity')
		plt.ylabel('correctness')

		# save figure
		if pth:
			tempfname = os.path.join(pth, 'weibull_fit_temp.png')
			plt.savefig(tempfname, dpi = 120)
			plt.close()
			return tempfname


def fit_weibull(db, i):
	take_last = min([i-15, 60])
	idx = np.array(np.arange(i-take_last+1, i+1), dtype = 'int')
	ifcorr = db.loc[idx, 'ifcorrect'].values.astype('int')
	opacit = db.loc[idx, 'opacity'].values.astype('float')

	# fit on non-nan
	notnan = ~(np.isnan(ifcorr))
	w.fit([1.0, 1.0])
	w = Weibull(opacit[notnan], ifcorr[notnan])
	return w

	
def set_opacity_if_fit_fails(corr, exp):
	mean_corr = np.mean(corr)
	if mean_corr > 0.8:
		exp['opacity'][0] *= 0.5
		exp['opacity'][1] *= 0.5 
		exp['opacity'][0] = np.max([exp['opacity'][0], 0.01])
	elif mean_corr < 0.6:
		exp['opacity'][0] = np.min([exp['opacity'][1]*2, 0.8])
		exp['opacity'][1] = np.min([exp['opacity'][1]*2, 1.0])


def correct_Weibull_fit(w, exp, newopac):
	# log weibul fit and contrast
	logs = []
	logs.append( 'Weibull params:  {} {}'.format( *w.params ) )
	logs.append( 'Contrast limits set to:  {0} - {1}'.format(*newopac) )

	# TODO this needs checking, removing duplicates and testing
	if newopac[1] <= newopac[0] or w.params[0] < 0 \
		or newopac[1] < 0.01 or newopac[0] > 1.0:

		set_opacity_if_fit_fails(w.orig_y, exp)
		logs.append( 'Weibull fit failed, contrast set to:  {0} - {1}'.format(*exp['opacity']) )
	else:
		exp['opacity'] = newopac

	# additional contrast checks
	precheck_opacity = list(exp['opacity'])
	if exp['opacity'][1] > 1.0:
		exp['opacity'][1] = 1.0
	if exp['opacity'][0] < 0.01:
		exp['opacity'][0] = 0.01
	if exp['opacity'][0] > exp['opacity'][1]:
		exp['opacity'][0] = exp['opacity'][1]/2

	if not (exp['opacity'] == precheck_opacity):
		logs.append('Opacity limits corrected to:  {0} - {1}'.format(*exp['opacity']))

	return exp, logs

def get_new_contrast(model, vmin=0.01, corr_lims=[0.52, 0.9], contrast_lims=None, method='random'):
	'''
	Methods:
	'5steps', '6steps', '12steps' - divides the contrast
	    range to equally spaced steps. The number of the
	    steps is defined at the beginning of the string.
	'6logsteps' - six logaritmic steps
	'4midlogsteps' - four log steps from the middle point
		of the contrast range in each direction (left and
			right)

	'''
	# TODO - maybe add 'random' to steps method
	#        this would shift the contrast points
	#        randomly from -0.5 to 0.5 of respective
	#        bin width (asymmteric in logsteps)
	if 'steps' in method:
		# get method details from string
		log = 'log' in method
		steps = ''
		for c in method:
			if c.isdigit():
				steps += c
			else:
				break
		steps = int(steps) if steps else 5

		# take contrast for specified correctness levels
		if not contrast_lims:
			if model.params[0] <= 0: # should be some small positive value
				contrast_lims = [vmin, vmin+0.2]
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
		steps = [0.1, 0.05, 0.01, 0.005]
		base_nonrep = not (len(check_contrast) == len(np.unique(check_contrast)))
		for s in steps:
			this_contrast = round2step(check_contrast, step=s)
			this_contrast = np.array([trim(x, vmin, 1.)
				for x in this_contrast])
			new_nonrep = len(this_contrast) == len(np.unique(this_contrast))
			if base_nonrep or new_nonrep:
				break
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
	low = map(lowfun, binval.index)
	high = map(highfun, binval.index)
	return binval, np.vstack([low, high]).T


def get_new_range(binval, bins, high_corr=0.8, low_corr=0.6):
	rng = np.where(binval > high_corr)[0]
	if np.any(rng):
		highval = bins[rng[0],0]
	else:
		highval = bins[-1,0]
	where_low = np.where(binval < low_corr)[0]
	if np.any(where_low):
		low = where_low[-1]
		lowval = bins[low, 0]
	else:
		go_below = bins[0,1] - bins[0,0]
		lowval = max(bins[0,0] - go_below*2, 0.03)
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
def fitw(df, ind, init_params=[1.,1.]):
    x = df.loc[ind, 'opacity'].values.astype('float64')
    y = df.loc[ind, 'ifcorrect'].values.astype('int32')
    w = Weibull(x, y)
    w.fit(init_params)
    return w


def idx_at(fit_num):
    current_trial = 45 + (fit_num-1)*10
    take_last = min([current_trial-15, 60])
    idx = np.array(np.arange(current_trial-take_last+1,
                             current_trial+1), dtype = 'int')
    return idx


def wfit_at(df, fit_num):
    idx = idx_at(fit_num)
    return fit_weibull(df, idx[-1])
