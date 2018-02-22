# all these functions are unused currently


# class for stepwise constrast adjustment
class Stepwise(object):
	"""Stepwise allows for simple staircase adjustment of
	a given parameter (in this exp - contrast).

	example
	-------
	To get a Stepwise object
	s = Stepwise(corr_ratio=[2, 1], step=0.1)"""
	def __init__(self, corr_ratio=[2,1], start=1.,
		step=0.1, vmin=0.1, vmax=1.):
		self.trial = 1
		self.direction = 'down'
		self.reversals = []
		self.rev_dir    = []
		self.corr_ratio = corr_ratio
		self.param = start
		self.step = step if isinstance(step, list) else [step]
		self.current_step = self.step[0]
		self.min = vmin
		self.max = vmax
		self.current_ratio = [0, 0]

	def add(self, resp):
		ind = 0 if resp else 1
		self.current_ratio[ind] += 1
		self.trial += 1
		self.check()

	def next(self):
		# first check whether param should change
		self.check()
		return self.param

	def check(self):
		if self.current_ratio[0] >= self.corr_ratio[0]:
			if self.direction == 'up':
				self.rev_dir.append(self.direction)
				self.reversals.append(self.param)
				self.direction = 'down'
			self.param -= self.current_step
			self.current_ratio = [0, 0]
		elif self.current_ratio[1] >= self.corr_ratio[1]:
			if self.direction == 'down':
				self.rev_dir.append(self.direction)
				self.reversals.append(self.param)
				self.direction = 'up'
			self.param += self.current_step
			self.current_ratio = [0, 0]

		self.param = trim(self.param, self.min, self.max)
		if len(self.step) > len(self.reversals):
			self.current_step = self.step[len(self.reversals)]


# from Weibull
# ------------
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


# for interactive plotting:
def fitw(df, ind=None, last=60, init_params=[1., 1.], method='Nelder-Mead'):
    if ind is None and last:
        n_rows = df.shape[0]
        start_ind = min([1, n_rows - last + 1])
        ind = np.arange(start_ind, n_rows + 1, dtype = 'int')

    # this might not be necessary
    x = df.loc[ind, 'opacity'].values.astype('float64')
    y = df.loc[ind, 'ifcorrect'].values.astype('int32')

    # fit on non-nan
    notnan = ~(np.isnan(y))
    w = Weibull(method=method)
    w.fit(x, y, init_params)
    return w


# - [ ] BaseMonkey needs fixing
class BaseMonkey(object):
    def __init__(self, respfun='random', response_mapping=None):
        if respfun == 'random':
            self.respfun = lambda x, c: sample([0, 1], 1)[0]
        self.response_mapping = response_mapping

    def respond(self, trial):
        correct_response = self.response_mapping(trial)
        response = self.respfun(trial, correct_response=correct_response)
        return response

# from exputils
# -------------
def scale_img(win, img, y_margin):
	winsize = win.size
	imsize = img.size
	if isinstance(imsize, np.ndarray) and imsize.ndim == 2:
		imsize = imsize[0, :]
	elif isinstance(imsize[0], list) and len(imsize[0]) == 2:
		imsize = imsize[0]
	imh_norm = imsize[1] / winsize[1]
	ypos = 1. - y_margin[0] - imh_norm
	low_margin = ypos - imh_norm
	ypos = int(ypos * winsize[1] / 2.)

	if low_margin < y_margin[1]:
		too_long_prop = (y_margin[1] - low_margin) / imh_norm
		img.size = np.array(imsize) * (1-too_long_prop)
		ypos = scale_img(win, img, y_margin)

	return ypos

