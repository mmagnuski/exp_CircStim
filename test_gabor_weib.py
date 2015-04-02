# -*- coding: utf-8 -*-

# add description

# TODOs:
# [x] test RT measurement on some platforms (timestamping
#         may not work...)
# [x] reset timer on stim presentation with callOnFlip
# [x] send LPT triggers with callOnFlip
# [ ] add logging to a file - useful in inspecting how fit
#     and overcoming fit failures works
# [ ] modularize and organize code
# [ ] use '\data' folder to save data
# [ ] test continue_dataframe for overwrite
# [x] add weibull fitting on fixed frame setup (1-1 frms)
# [ ] add simple instructions     (!)
# [ ] add simple training         (!)
#     -> +feedback
#     -> (before) +slowdown +full-contrast?
# 
# not necessary:
# [ ] ? present some scatter feedback
# [ ] load matplotlib and seaborn (here or in stimutils)
# [ ] load seaborn conditionally
# [ ] ? check interpolate = True in visual.ImageStim

# imports
print 'importing psychopy...'
from psychopy import visual, core, event
from exputils import getFrameRate, ms2frames, getUserName, continue_dataframe, Weibull, plot_Feedback
from random   import randint, uniform #, choice
from ctypes   import windll
import os
import numpy  as np
import pandas as pd


# experiment settings
exp = {}
exp['clock']       = core.Clock()
exp['participant'] = getUserName(intUser = False)
exp['debug']       = True
exp['use trigger'] = False
exp['port address'] = None
exp['break after'] = 15 # how often subjects have a break

exp['targetTime']  = [1]
exp['SMI']         = [2] # Stimulus Mask Interval
exp['fixTimeLim']  = [0.75, 2.5]
exp['maskTime']    = [20]
exp['opacity']     = [0.05, 0.8]
exp['orientation'] = [0, 45, 90, 135]
exp['corrLims']    = [0.55, 0.9]

exp['use keys']    = ['f', 'j']
exp['respWait']    = 1.5

resp = {}
choose_resp = randint(0, 1)
resp[0]   = exp['use keys'][choose_resp]
resp[90]  = exp['use keys'][choose_resp]
resp[45]  = exp['use keys'][1 - choose_resp]
resp[135] = exp['use keys'][1 - choose_resp]
exp['keymap'] = resp
exp['choose_resp'] = choose_resp

# port settings
portdict = {}
portdict['send'] = exp['use trigger']
portdict['port address'] = exp['port address']
portdict['codes'] = {'fix' : 1, 'mask' : 2}
portdict['codes'].update({'target_'+str(ori) : 4+i \
	for i,ori in enumerate(exp['orientation'])})


# change below to logging:
print 'keymap: ', exp['keymap']

# create a window
# ---------------
win = visual.Window([800,600],monitor="testMonitor", 
    units="deg", color = [-0.2, -0.2, -0.2], fullscr=True)
# get fame rate
frm = getFrameRate(win)
win.setMouseVisible(False)

# get path
pth   = os.path.dirname(os.path.abspath(__file__))
ifcnt = continue_dataframe(pth, exp['participant'] + '.xls')
exp['path'] = pth

if not ifcnt:
	# create DataFrame
	# ----------------
	# define column names:
	colNames = ['time', 'fixTime', 'targetTime', 'SMI', \
				'maskTime', 'opacity', 'orientation', \
				'response', 'ifcorrect', 'RT']
	# CHANGE - dtypes are not being used
	dtypes   = ['float', 'float', 'int', 'int', \
				'int', 'float', 'int', 'str', 'int', 'float']
	# combined columns
	cmb = ['targetTime', 'SMI', 'maskTime', 'orientation']
	# how many times each combination should be presented
	perComb = 140;
	# generate trial combinations
	lst = [
			[i, j, l, m ] \
				for i in exp['targetTime']  \
				for j in exp['SMI']         \
				for l in exp['maskTime']    \
				for m in exp['orientation'] \
			if i + j < 5] \
		* perComb

	# turn to array and shuffle rows
	lst = np.array(lst)
	np.random.shuffle(lst)

	# construct data frame
	shp = lst.shape
	db = pd.DataFrame(
		index = np.arange(1, shp[0] + 1),
		columns = colNames
		)
	exp['numTrials'] = len(db)

	# fill the data frame from lst
	for i, r in enumerate(cmb):
		db[r] = lst[:, i]

	# add fix time in frames
	db['fixTime'] = ms2frames( 
		np.random.uniform(
			low = exp['fixTimeLim'][0], 
			high = exp['fixTimeLim'][1], 
			size = exp['numTrials']
			) * 1000,
		frm['time']
		)

	startTrial = 1
else:
	db, startTrial = ifcnt
	exp['numTrials'] = len(db)


# ==def for gabor creation==
def gabor(win = win, ori = 0, opa = 1.0, 
		  pos  = [0, 0], size = 7, 
		  units = 'deg', sf = 1.5):
	return visual.GratingStim(win     = win,  mask = "gauss", \
							  size    = size, pos  = pos, \
							  sf      = sf,   ori  = ori,      \
							  opacity = opa,  units = units)

# prepare stimuli
stim = {}
stim['window'] = win
stim['target'] = gabor()
# CHANGE - window size, so that it is accurate...
stim['centerImage'] = visual.ImageStim(win, image=None,  
            pos=(0.0, 0.0), size=(14*80,6*80), units = 'pix')


# mask - all gabor directions superposed
mask_ori = [0, 45, 90, 135]
stim['mask'] = []

for o in mask_ori:
	stim['mask'].append(gabor(ori = o, opa = 0.25))

# create fixation cross:
v = np.array(
		[
			[0.1, -1], 
			[0.1, 1],
			[-0.1, 1],
			[-0.1, -1]
		]
	)

def whiteshape(v, win = win):
	return visual.ShapeStim(
		win, 
		lineWidth  = 0.5, 
		fillColor  = [1, 1, 1], 
		lineColor  = [1, 1, 1], 
		vertices   = v, 
		closeShape = True
		)

fix = []
fix.append(whiteshape(v))
fix.append(whiteshape(
	np.fliplr(v)
	))
stim['fix'] = fix
del fix

def onflip_work(portdict, code='', clock=None):
	if clock:
		clock.reset()
	if portdict['send'] and code:
		windll.inpout32.Out32(portdict['port address'], 
			portdict['codes'][code])

# we do not need to clear port on the new amplifier
# so this short def is 'just-in-case'
def clear_port(portdict):
	windll.inpout32.Out32(portdict['port address'], 0)

def present_trial(tr, exp = exp, stim = stim, db = db, win = win):

	# PREPARE
	# -------
	db.loc[tr, 'opacity'] = np.round(
		np.random.uniform(
			low = exp['opacity'][0], 
			high = exp['opacity'][1], 
			size = 1
			)[0], decimals = 3)
	# set target properties (takes less than 1 ms)
	stim['target'].ori = db.loc[tr]['orientation']
	stim['target'].opacity = db.loc[tr]['opacity']
	target_code = 'target_' + str(db.loc[tr]['orientation'])

	# get trial start time (!)
	db.loc[tr, 'time'] = core.getTime()

	# present fix:
	win.callOnFlip(onflip_work, portdict, code='fix')
	for f in np.arange(db.loc[tr]['fixTime']):
		for fx in stim['fix']:
			fx.draw()
		win.flip()

	# present target
	win.callOnFlip(onflip_work, portdict, code=target_code,
		clock=exp['clock'])
	for f in np.arange(db.loc[tr]['targetTime']):
		stim['target'].draw()
		win.flip()

	# interval
	for f in np.arange(db.loc[tr]['SMI']):
		win.flip()

	# mask
	win.callOnFlip(onflip_work, portdict, code='mask')
	for f in np.arange(db.loc[tr]['maskTime']):
		for m in stim['mask']:
			m.draw()
		win.flip()

	# RESPONSE
	# --------
	# check if response
	k = event.getKeys(keyList = exp['use keys'] + ['q'], 
					  timeStamped = exp['clock'])
	
	# wait for response (timeout)
	if not k:
		k = event.waitKeys(maxWait = exp['respWait'], 
					       keyList = exp['use keys'] + ['q'], 
					       timeStamped = exp['clock'])
	
	# calculate RT and ifcorrect
	if k:
		key, RT = k[0]

		# if debug - test for quit
		if exp['debug'] and key == 'q':
				core.quit()

		# performance
		db.loc[tr, 'response']  = key
		db.loc[tr, 'RT']        = RT
		db.loc[tr, 'ifcorrect'] = int(exp['keymap']\
									  [db.loc[tr]['orientation']] == key)
	else:
		db.loc[tr, 'response']  = 'NoResp'
		db.loc[tr, 'ifcorrect'] = 0

def show_resp_rules(exp = exp, win = win):

	# create diagonal on one side and cardinal on the other
	ch    = exp['choose_resp']

	stims = []
	ornt      = [[0, 90], [45, 135]]
	ornt      = ornt[ch] + ornt[1- ch]

	positions = [[-1, -0.5], [-1, 0.5], \
				 [1, -0.5],  [1, 0.5]]
	positions = np.array(positions) * 10

	for o, p in zip(ornt, positions):
		stims.append(gabor(ori = o, pos  = p, size = 4, 
						   units = 'deg'))

	# add info on correct responses
	tx = [
		u'naciśnij  ' + exp['keymap'][ornt[0]], 
		u'naciśnij  ' + exp['keymap'][ornt[2]],
		u'aby przejść dalej naciśnij spację'
		]
	positions = [
		[-0.5, 0.75], 
		[0.5, 0.75],
		[0, -0.85]
		]
	txStim = []

	for t, p in zip(tx, positions):
		txStim.append(visual.TextStim(win, text = t, pos = p, units = 'norm'))

	# draw all:
	for t in txStim:
		t.draw()

	for g in stims:
		g.draw()

	win.flip()

	# wait for space:
	k = event.waitKeys(keyList = ['space'])


def present_break(t, exp = exp, win = win):
	tex  = u'Ukończono  {0} / {1}  powtórzeń.\nMożesz teraz ' + \
		   u'chwilę odetchnąc.\nNaciśnij spację aby kontynuowac...'
	tex  = tex.format(t, exp['numTrials'])
	info = visual.TextStim(win, text = tex, pos = [0, 0], units = 'norm')
	
	info.draw()
	win.flip()

	# wait for space key:
	k = event.waitKeys(keyList = ['space', 'escape', 'return'])


def set_opacity_if_fit_fails(corr, exp):
	mean_corr = np.mean(corr)
	if mean_corr > 0.8:
		exp['opacity'][0] *= 0.5
		exp['opacity'][1] *= 0.5 
		exp['opacity'][0] = np.max([exp['opacity'][0], 0.01])
	elif mean_corr < 0.6:
		exp['opacity'][0] = np.min([exp['opacity'][1]*2, 0.8])
		exp['opacity'][1] = np.min([exp['opacity'][1]*2, 1.0])

def fit_weibull(db, i):
#	if i < 40:
#		idx = np.array(np.linspace(1, i, num = i), dtype = 'int')
#	else:
#		idx = np.array(np.linspace(i-40+1, i, num = 40), dtype = 'int')
	idx = np.array(np.linspace(1, i, num = i), dtype = 'int')
	ifcorr = db.loc[idx, 'ifcorrect'].values.astype('int')
	opacit = db.loc[idx, 'opacity'].values.astype('float')

	# fit on non-nan
	notnan = ~(np.isnan(ifcorr))
	w = Weibull(opacit[notnan], ifcorr[notnan])
	w.fit([1.0, 1.0])
	return w

# show response rules:
show_resp_rules()

# show trials:
for i in range(startTrial, exp['numTrials'] + 1):
	present_trial(i)
	win.flip()

	# present break 
	if (i) % exp['break after'] == 0:
		# save data before every break
		db.to_excel( exp['participant'] + '.xls')
		
		# if break was within first 100 trials,
		# fit Weibull function
		if i < 101:
			w = fit_weibull(db, i)
			print 'Weibull params: ', w.params
			
			newopac = w._dist2corr(exp['corrLims'])
			if newopac[1] < 0.005 or newopac[1] <= newopac[0] or w.params[0] < 0 or newopac[1] < 0.01 or newopac[0] > 1.0:
				set_opacity_if_fit_fails(w.orig_y, exp)
			else:
				exp['opacity'] = newopac
            
			if exp['opacity'][1] > 1.0:
				exp['opacity'][1] = 1.0
			if exp['opacity'][0] < 0.01:
				exp['opacity'][0] = 0.01
			if exp['opacity'][0] > exp['opacity'][1]:
				exp['opacity'][0] = exp['opacity'][1]/2
                

			# DEBUG
			print 'opacity limits set to: ', exp['opacity']

			# show weibull fit
			datapth = os.path.join(exp['path'], r'data')
			plot_Feedback(stim, w, datapth)

		# break and refresh keyboard mapping
		present_break(i)
		show_resp_rules()

	win.flip()
	core.wait(0.5) # change to variable intertrial interval?

db.to_excel( exp['participant'] + '.xls')
core.quit()