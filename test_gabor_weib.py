# -*- coding: utf-8 -*-

# add description

# TODOs:
# [ ] add simple instructions     (!)
# [ ] add simple training         (!)
#     -> +feedback
#     -> (before) +slowdown +full-contrast?
# [ ] remove print statements and add logging to a file - 
#     useful in inspecting how fit and overcoming fit 
#     failures works
# [ ] test continue_dataframe for overwrite
# [ ] modularize and organize code
# [x] use '\data' folder to save data
# [x] test RT measurement on some platforms (timestamping
#         may not work...)
# [x] reset timer on stim presentation with callOnFlip
# [x] send LPT triggers with callOnFlip
# [x] add weibull fitting on fixed frame setup (1-1 frms)
# 
# not necessary:
# [ ] load seaborn conditionally
# [ ] ? present some scatter feedback
# [ ] ? check interpolate = True in visual.ImageStim

# imports
print 'importing psychopy...'
from psychopy  import visual, core, event
from exputils  import getFrameRate, ms2frames, getUserName, continue_dataframe, Weibull, plot_Feedback
from random    import randint, uniform #, choice
from ctypes    import windll
from stimutils import stim, gabor
import os
import numpy  as np
import pandas as pd


# experiment settings
exp = {}
exp['debug']       = True
exp['clock']       = core.Clock()
exp['use trigger'] = True
exp['port address'] = '0xDC00' # string, for example '0xD05'
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
portdict['port address'] = int(exp['port address'], base=16) \
						   if exp['port address'] and portdict['send'] \
						   else exp['port address']
portdict['codes'] = {'fix' : 1, 'mask' : 2}
portdict['codes'].update({'target_'+str(ori) : 4+i \
						   for i,ori in enumerate(exp['orientation'])
						   })


# change below to logging:
print 'keymap: ', exp['keymap']
exp['participant'] = getUserName(intUser = False)


# get path
pth   = os.path.dirname(os.path.abspath(__file__))
exp['path'] = pth
# ensure 'data' directory is available:
exp['data'] = os.path.join(pth, 'data')
if not os.path.isdir(exp['data']):
	os.mkdir(exp['data'])

# get frame rate:
# get fame rate
frm = getFrameRate(stim['window'])

# check if continue with previous dataframe:
ifcnt = continue_dataframe(exp['data'], exp['participant'] + '.xls')

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

def present_trial(tr, exp = exp, stim = stim, db = db, 
					  win = stim['window']):

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

def show_resp_rules(exp = exp, win = stim['window']):

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


def present_break(t, exp = exp, win = stim['window']):
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
	stim['window'].flip()

	# present break 
	if (i) % exp['break after'] == 0:
		# save data before every break
		db.to_excel(os.path.join(exp['data'], exp['participant'] + '.xls'))
		
		# if break was within first 100 trials,
		# fit Weibull function
		if i < 101:
			w = fit_weibull(db, i)
			print 'Weibull params: ', w.params
			
			newopac = w._dist2corr(exp['corrLims'])
			# TODO this needs checking, removing duplicates and testing
			if newopac[1] < 0.005 or newopac[1] <= newopac[0] or w.params[0] < 0 \
				or newopac[1] < 0.01 or newopac[0] > 1.0:
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
			plot_Feedback(stim, w, exp['data'])

		# break and refresh keyboard mapping
		present_break(i)
		show_resp_rules()

	stim['window'].flip()
	core.wait(0.5) # pre-fixation time is always the same

db.to_excel(os.path.join(exp['data'], exp['participant'] + '.xls'))
core.quit()