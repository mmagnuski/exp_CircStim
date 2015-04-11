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


# ==def for gabor creation==
def gabor(win = win, ori = 0, opa = 1.0, 
		  pos  = [0, 0], size = 7, 
		  units = 'deg', sf = 1.5):
	return visual.GratingStim(win     = win,  mask = "gauss", \
							  size    = size, pos  = pos, \
							  sf      = sf,   ori  = ori,      \
							  opacity = opa,  units = units)

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


# stimuli presentation etc.
# -------------------------

# get frame rate 
# (this is now done in settings, maybe no need to repeat?):
# exp['frm'] = getFrameRate(stim['window'])

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
	# check if response
	k = event.getKeys(keyList = exp['use keys'] + ['q'], 
					  timeStamped = exp['clock'])
	
	# wait for response (timeout)
	if not k:
		k = event.waitKeys(maxWait = exp['respWait'], 
					       keyList = exp['use keys'] + ['q'], 
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
		db.loc[tr, 'ifcorrect'] = int(exp['keymap']\
									  [db.loc[tr]['orientation']] == key)
	else:
		db.loc[tr, 'response']  = 'NoResp'
		db.loc[tr, 'ifcorrect'] = 0


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


def txt(win=win, **kwargs):
	return visual.TextStim(win, **kwargs)


class Instructions:

	nextpage = 0
	navigation = {'leftarrow': 'prev',
				  'rightarrow': 'next'}

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
			self.create_page(self.nextpage)
			# draw page elements
			for it in self.pageitems:
				it.draw()
			self.win.flip()

			# wait for response
			k = event.waitKeys(self.navigation.keys())
			action = self.navigation[k]

			# go next/prev according to the response
			if action == 'next':
				self.nextpage += 1
			else:
				self.nextpage -= 1
				self.nextpage = max(0, self.nextpage)

	def create_page(self, page_num=self.nextpage):
		self.pageitems = [self.parse_item(i) for i in self.page(page_num)]

	def parse_item(self, item):
		# currently: gabor or text:
		mapdict = {'gabor': self.create_gabor,
			'text': self.create_text}
		fun = mapdict.get(item['item'], [])
		if fun:
			return fun(item)

	def create_gabor(self, item):
		return gabor(**item['value'])

	def create_text(self, item):
		return txt(**item['value'])