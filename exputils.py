from psychopy       import gui
from psychopy       import event
from matplotlib     import pyplot   as plt
from scipy.optimize import minimize
from PIL            import Image
import numpy  as np
import pandas as pd
import os
import re

# TODOs:
# [ ] - continue_dataframe should test for overwrite
#
# Weibull class:
# [ ] - refactor for ease of use 
#       -> do not have to pass additional args 
#          if they are in the model 
#       -> but allow to pass these args if needed
# [ ] - ! ensure that orig_y and y are copies !
# [ ] - predict (now performed by _fun but could change names)
# [ ] - ! think about adding params to to predictors for both slope and position


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
	
	def plot(self, pth):
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
		tempfname = os.path.join(pth, 'weibull_fit_temp.png')
		plt.savefig(tempfname, dpi = 120)
		plt.close()
		return tempfname


def plot_Feedback(stim, plotter, pth):
    # get file names from 
    imfls = plotter.plot(pth)
    if not isinstance(imfls, type([])):
    	imfls = [imfls]

	for im in imfls:
		# check image size:
		img = Image.open(im)
		imgsize = np.array(img.size)
		del img

		# set image
		stim['centerImage'].size = np.round(imgsize * 0.8)
		stim['centerImage'].setImage(im)
		stim['centerImage'].draw()
		stim['window'].update()

		k = event.getKeys()
		resp = False

		while not resp:
			resp = event.getKeys()

def getFrameRate(win, frames = 25):
	# get frame rate
	print "testing frame rate..."
	frame = {}
	frame['rate'] = win.getActualFrameRate(nIdentical = frames)
	frame['time'] = 1000.0 / frame['rate']
	print "frame rate: " + str(frame['rate'])
	print "time per frame: " + str(frame['time'])
	return frame


def ms2frames(times, frame_time):
    
	tp = type(times)
	if tp == type([]):
	    frms = []
	    for t in times:
	        frms.append( int( round(t / frame_time) ) )
	elif tp == type({}):
	    frms = {}
	    for t in times.keys():
	        frms[t] = int( round(times[t] / frame_time) )
	elif tp == type(np.array([1])):
		frms = np.array(
			np.round(times / frame_time),
			dtype = int
			)
	else:
		frms = [] # or throw ValueError

	return frms


def fillz(val, num):
    '''
    fillz(val, num)
    exadds zero to the beginning of val so that length of
    val is equal to num. val can be string, int or float
    '''
    
    # if not string - turn to a string
    if not isinstance(val, basestring):
        val = str(val)
     
    # add zeros
    ln = len(val)
    if ln < num:
        return '0' * (num - ln) + val
    else:
        return val


# get user name:
def getUserName(intUser = True):
    '''
    PsychoPy's simple GUI for entering 'stuff'
    Does not look nice or is not particularily user-friendly
    but is simple to call.
    Here it just asks for subject's name/code
    and returns it as a string
    '''
    myDlg = gui.Dlg(title="Pseudonim", size = (800,600))
    myDlg.addText('Podaj numer osoby badanej')
    myDlg.addField('numer')
    myDlg.show()  # show dialog and wait for OK or Cancel

    if myDlg.OK:  # the user pressed OK
        dialogInfo = myDlg.data
        if intUser:
            try:
                user = int(dialogInfo[0])
            except (ValueError, TypeError):
                user = None
        else:
            user = dialogInfo[0]
    else:
        user = None
   
    return user

def continue_dataframe(pth, fl):
	
	# test if file exists:
	flfl = os.path.join(pth, fl)
	ifex = os.path.isfile(flfl)

	if not ifex:
		return ifex
	else:
		# load with pandas and take first column
		df = pd.read_excel(flfl)
		col = df.columns
		col = df[col[0]]

		# test first col for nonempty
		isempt, = np.nonzero(np.isnan(col))
		if isempt.any():
			# return dataframe and first trial to resume from:
			return df, isempt[0] + 1
		else:
			# here it should return something special 
			# not to everwrite the file (becasue file
			# absent and file presnt but finished re-
			# sponses are identical)
			return False 


def free_filename(pth, subj, givenew = True):

	# list files within the dir
	fls = os.listdir(pth)
	n_undsc_in_subj = 0
	ind = -1
	while True:
		try:
			ind = subj['symb'][ind+1:-1].index('_')
			n_undsc_in_subj += 1
		except (IndexError, ValueError):
			break

	# predefined file pattern
	subject_pattern = subj['symb'] + '_' + r'[0-9]+'

	# check files for pattern
	reg = re.compile(subject_pattern)
	used_files = [reg.match(itm).group() for ind, itm in enumerate(fls) \
		if not (reg.match(itm) == None)]
	used_files_int = [int(f.split('_')[1 + n_undsc_in_subj][0:2]) for f in used_files]

	if givenew:
		if len(used_files) == 0:
			# if there are no such files - return prefix + S01
			return 1, subj['symb'] + '_01'
		else:
			# check max used number:
			mx = 1
			for temp in used_files_int:
				if temp > mx:
					mx = temp

			return mx + 1, subj['symb'] + '_' + fillz(mx + 1, 2)
	else:
		return not subj['ind'] in used_files_int
