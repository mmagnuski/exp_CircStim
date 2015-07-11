# weibull module for fitting Weibull psychometric function

# TODOs:
# Weibull class:
# [ ] - refactor for ease of use 
#       -> do not have to pass additional args 
#          if they are in the model 
#       -> but allow to pass these args if needed
# [ ] - ! ensure that orig_y and y are copies !
# [ ] - predict (now performed by _fun but could change names)
# [ ] - ! think about adding params to to predictors for both slope and position

# imports
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import numpy as np
import os


class Weibull:
	'''
	Weibull is a simple class for fitting and
	evaluating Weibull function.

	example:
	--------
	w = Weibull(stim_intensity, if_resp_correct)
	initparams = [1., 1.]
	w.fit(initparams)
	final_params = w.params
	'''

	def __init__(self, x, y):
		self.x = x
		# y is 0 or 1 - this is problematic for log
		self.orig_y = y # CHECK ensure that this is a copy!
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

	def inverse(self, corrinput):
		invfun = lambda cntr: (corrinput - self._fun(self.params, cntr))**2
		# optimize with respect to correctness
		return minimize(invfun, 1.0, method='Nelder-Mead')['x'][0]
	
	def _dist2corr(self, corr):
		return map(self.inverse, corr)
	
	def plot(self, pth=''):
		# get predicted data
		numpnts = 1000
		x = np.linspace(0., 1., num = numpnts)
		y = self._fun(self.params, x)

		# add noise to data to increase visibility
		l = len(self.x)
		yrnd = np.random.uniform(-0.05, 0.05, l)

		# plot
		plt.hold(True)
		plt.grid()
		plt.plot(x, y, zorder = 1)
		plt.scatter(self.x, self.orig_y + yrnd, alpha=0.6, lw=0, c=[0.3, 0.3, 0.3])

		# aesthetics
		maxval = np.max(self.x)
		uplim = np.round(maxval + 0.5, decimals = 1)
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
	w = Weibull(opacit[notnan], ifcorr[notnan])
	w.fit([1.0, 1.0])
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
	if newopac[1] < 0.005 or newopac[1] <= newopac[0] or w.params[0] < 0 \
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
