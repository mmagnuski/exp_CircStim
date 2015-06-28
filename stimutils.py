# -*- coding: utf-8 -*-

from psychopy import core, visual, event, monitors
from ctypes    import windll
from settings import exp, db, startTrial
from exputils  import getFrameRate
import numpy  as np
import yaml


# import monitor settings
monitorName = "testMonitor"
monitors = monitors.getAllMonitors()
if "BENQ-XL2411" in monitors:
	monitorName = "BENQ-XL2411"

# create a window
# ---------------
win = visual.Window(monitor=monitorName, 
    units="deg", color = [-0.2, -0.2, -0.2], fullscr=True)
win.setMouseVisible(False)


# STIMULI
# -------

def txt(win=win, **kwargs):
	return visual.TextStim(win, units='norm', **kwargs)

def txt_newlines(win=win, text='', **kwargs):
	text = text.replace('\\n', '\n')
	return visual.TextStim(win, text=text, units='norm', **kwargs)

# gabor creation
def gabor(win = win, ori = 0, opa = 1.0, 
		  pos  = [0, 0], size = exp['gabor size'],
		  units = 'deg', sf = exp['gabor freq']):
	return visual.GratingStim(win     = win,  mask = "gauss", \
							  size    = size, pos  = pos, \
							  sf      = sf,   ori  = ori,      \
							  opacity = opa,  units = units)

# TODO should units be set to 'deg' here?
def whiteshape(v, win = win):
	return visual.ShapeStim(
		win, 
		lineWidth  = 0.5, 
		fillColor  = [1, 1, 1], 
		lineColor  = [1, 1, 1], 
		vertices   = v, 
		closeShape = True
		)

# create fixation cross:
def fix():
	v = np.array(
			[
				[0.1, -1], 
				[0.1, 1],
				[-0.1, 1],
				[-0.1, -1]
			]
		)

	fix = []
	fix.append(whiteshape(v))
	fix.append(whiteshape(
		np.fliplr(v)
		))
	return fix

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

stim['fix'] = fix()

# feedback circle:
stim['feedback'] = visual.Circle(stim['window'], radius=2.5,
	edges=32, units='deg')


# TRIGGERS
# --------

# def for onflip clock reset and port trigger
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


# PRESENTATION
# ------------

def present_trial(tr, exp = exp, stim = stim, db = db, 
					  win = stim['window']):
	# PREPARE
	# -------

	# randomize opacity if not set
	if exp['opacity'][0] == exp['opacity'][1]:
		db.loc[tr, 'opacity'] = exp['opacity'][0]
	else:
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

	# get trial start time
	db.loc[tr, 'time'] = core.getTime()


	# PRESENT
	# -------

	# present fix:
	win.callOnFlip(onflip_work, exp['port'], code='fix')
	for f in np.arange(db.loc[tr]['fixTime']):
		for fx in stim['fix']:
			fx.draw()
		win.flip()

	# present target
	win.callOnFlip(onflip_work, exp['port'], code=target_code,
		clock=exp['clock'])
	for f in np.arange(db.loc[tr]['targetTime']):
		stim['target'].draw()
		win.flip()

	# interval
	for f in np.arange(db.loc[tr]['SMI']):
		win.flip()

	# mask
	win.callOnFlip(onflip_work, exp['port'], code='mask')
	for f in np.arange(db.loc[tr]['maskTime']):
		for m in stim['mask']:
			m.draw()
		win.flip()


	# RESPONSE
	# --------

	# which keys we wait for:
	keys = exp['use keys']
	if exp['debug']: keys += ['q']

	# check if response
	k = event.getKeys(keyList = keys, 
					  timeStamped = exp['clock'])

	# wait for response (timeout)
	if not k:
		k = event.waitKeys(maxWait = exp['respWait'], 
					       keyList = keys, 
					       timeStamped = exp['clock'])
		# TODO - send response marker?

	# calculate RT and ifcorrect
	if k:
		key, RT = k[0]

		# if debug - test for quit
		if exp['debug'] and key == 'q':
				core.quit()

		# performance
		db.loc[tr, 'response']  = key
		db.loc[tr, 'RT']        = RT
		target_ori = db.loc[tr]['orientation']
		db.loc[tr, 'ifcorrect'] = int(exp['keymap'][target_ori] == key)
	else:
		db.loc[tr, 'response']  = 'NoResp'
		db.loc[tr, 'ifcorrect'] = 0


def present_feedback(i, db=db, stim=stim):
	if db.loc[i, 'ifcorrect'] == 1:
		stim['feedback'].setFillColor([0.1, 0.9, 0.1])
	else:
		stim['feedback'].setFillColor([0.9, 0.1, 0.1])

	for f in range(0, exp['fdb time'][0]):
		stim['feedback'].draw()
		stim['window'].flip()


# class for stepwise constrast adjustment
class Stepwise(object):
	"""Stepwise allows for simple staircase adjustment of
	a given parameter.
	example
	-------
	To get a Stepwise object
	s = Stepwise([2, 1], 0.1)"""
	def __init__(self, corr_ratio=[2,1], start=1.,
		step=0.1, min=0.1, max=1., start_trial=1):
		self.trial = start_trial
		self.corr_ratio = corr_ratio
		self.param = start
		self.step = step
		self.min = min
		self.max = max
		self.current_ratio = [0, 0]

	def add(self, resp):
		ind = 0 if resp else 1
		self.current_ratio[ind] += 1
		self.check()

	def next(self):
		# first check whether param should change
		self.check()
		return self.param

	def check(self):
		if self.current_ratio[0] > self.corr_ratio[0]:
			self.param += self.step
			self.current_ratio = [0, 0]
		elif self.current_ratio[1] > self.corr_ratio[1]:
			self.param -= self.step
			self.current_ratio = [0, 0]

		self.param = trim(self.param, self.min, self.max)


# instructions etc.
# -----------------

# this should go to instructions module
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


def give_training_db(db, exp=exp, slowdown=8):
	# copy the dataframe
	train_db = db.copy()
	# shuffle orientations
	val = train_db.loc[:, 'orientation'].values
	np.random.shuffle(val)
	train_db.loc[:, 'orientation'] = val

	# change targetTime, SMI (and maybe maskTime?)
	train_db.loc[:, 'targetTime'] = exp['targetTime'][0] * slowdown
	train_db.loc[:, 'SMI'] = exp['SMI'][0] * slowdown
	return train_db


class Instructions:

	nextpage   = 0
	mapdict    = {'gabor': gabor, 'text': txt_newlines}
	navigation = {'left': 'prev',
				  'right': 'next',
				  'space': 'next'}

	def __init__(self, fname, win=win):
		self.win = win

		# get instructions from file:
		with open(fname, 'r') as f:
		    instr = yaml.load_all(f)
		    self.pages = list(instr)
		self.stop_at_page = len(self.pages)

	def present(self, start=None, stop=None):
		if not isinstance(start, int):
			start = self.nextpage
		if not isinstance(stop, int):
			stop = len(self.pages)

		# show pages:
		self.nextpage = start
		while self.nextpage < stop:
			# create page elements
			self.create_page()
			# draw page elements
			for it in self.pageitems:
				it.draw()
			self.win.flip()

			# wait for response
			k = event.waitKeys(keyList=self.navigation.keys())[0]
			action = self.navigation[k]

			# go next/prev according to the response
			if action == 'next':
				self.nextpage += 1
			else:
				self.nextpage = max(0, self.nextpage - 1)

	def create_page(self, page_num=None):
		if not isinstance(page_num, int):
			page_num = self.nextpage
		self.pageitems = [self.parse_item(i) for i in self.pages[page_num]]

	def parse_item(self, item):
		# currently: gabor or text:
		fun = self.mapdict.get(item['item'], [])

		if fun: return fun(**item['value'])
