# -*- coding: utf-8 -*-
# shebang with env\python etc. ? 

# this is a simple test for first grant experiment
# (gabor brightness)

# TODOs:
# [x] add fixation cross          (!)
# [x] variable wait time on fixation (0.75 - 2.5)
# [x] test if responses are presented ok in instructions (!)
# [x] add keypresses (or NaN, 'noResp' ?) to database
# [x] add clock time to database
# [x] continue dataframe
# [ ] test continue_dataframe for overwrite
# [ ] add simple instructions     (!)
# [ ] add simple training         (!)
# [ ] present some scatter feedback
# [ ] check interpolate = True in visual.ImageStim
# [ ] variable inter-trial time?

# imports
print 'importing psychopy...'
from psychopy import visual, core, event
from exputils import getFrameRate, ms2frames, getUserName, continue_dataframe
from random   import randint, uniform #, choice
import os
import numpy  as np
import pandas as pd


# experiment settings
exp = {}
exp['participant'] = getUserName(intUser = False)
exp['debug']       = True
exp['use trigger'] = False
exp['break after'] = 15 # how often subjects have a break

exp['targetTime']  = [1, 2, 3]
exp['SMI']         = [1, 2, 3] # Stimulus Mask Interval
exp['fixTimeLim']  = [0.75, 2.5]
exp['maskTime']    = [20]
exp['opacity']     = [0.05, 0.1, 0.2, 0.4]
exp['orientation'] = [0, 45, 90, 135]

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

print 'keymap: ', exp['keymap']

# create a window
# ---------------
win = visual.Window([800,600],monitor="testMonitor", 
    units="deg", color = [-0.2, -0.2, -0.2], fullscr=True)
# get fame rate
frm = getFrameRate(win)
win.mouseVisible = False



# get path
pth   = os.path.dirname(os.path.abspath(__file__))
ifcnt = continue_dataframe(pth, exp['participant'] + '.xls')

if not ifcnt:
	# create DataFrame
	# ----------------
	# define column names:
	colNames = ['time', 'fixTime', 'targetTime', 'SMI', \
				'maskTime', 'opacity', 'orientation', \
				'response', 'ifcorrect', 'RT']
	# combined columns
	cmb = ['targetTime', 'SMI', 'opacity', 'maskTime', 'orientation']
	# how many times each combination should be presented
	perComb = 3;
	# generate trial combinations
	lst = [
			[i, j, k, l, m ] \
				for i in exp['targetTime']  \
				for j in exp['SMI']         \
				for k in exp['opacity']     \
				for l in exp['maskTime']    \
				for m in exp['orientation'] \
			] \
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
stim['target'] = gabor()

mask_ori = [0, 45, 90, 135]
stim['mask'] = []

# mask is more complex story: 
# all gabor directions superposed
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

# change the procedure for presenting 
# stimuli so that first frame could send
# trigger and record time

def present_trial(tr, exp = exp, stim = stim, db = db, win = win):

	# set target properties (takes less than 1 ms)
	# t1 = core.getTime()
	stim['target'].ori = db.loc[tr]['orientation']
	stim['target'].opacity = db.loc[tr]['opacity']
	# t_full = core.getTime() - t1
	# print 'changing stimuli prprts took {0} seconds'.format(t_full)

	# get trial start time (!)
	db.loc[tr, 'time'] = core.getTime()

	# present fix:
	for f in np.arange(db.loc[tr]['fixTime']):
		for fx in stim['fix']:
			fx.draw()
		win.flip()

	# get time (should be taken on first frame)
	t1 = core.getTime()
	for f in np.arange(db.loc[tr]['targetTime']):
		stim['target'].draw()
		win.flip()

	# interval
	for f in np.arange(db.loc[tr]['SMI']):
		win.flip()

	# mask
	for f in np.arange(db.loc[tr]['maskTime']):
		for m in stim['mask']:
			m.draw()
		win.flip()	

	# check if response
	k = event.getKeys(keyList = exp['use keys'] + ['q'], 
					  timeStamped = True)
	
	# wait for response (timeout)
	if not k:
		k = event.waitKeys(maxWait = exp['respWait'], 
					       keyList = exp['use keys'] + ['q'], 
					       timeStamped = True)
	
	# calculate RT and ifcorrect
	if k:
		key = k[0][0]
		t2  = k[0][1]

		# if debug - test for quit
		if exp['debug']:
			if key == 'q':
				core.quit()

		# performance
		db.loc[tr, 'response']  = key
		db.loc[tr, 'RT']        = t2 - t1
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
		present_break(i)

	core.wait(1) # change to variable intertrial interval