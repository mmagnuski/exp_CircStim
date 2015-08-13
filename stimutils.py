# -*- coding: utf-8 -*-

# monkey-patch pyglet shaders:
# ----------------------------
# fragFBOtoFramePatched = '''
#     uniform sampler2D texture;

#     float rand(vec2 seed){
#         return fract(sin(dot(seed.xy ,vec2(12.9898,78.233))) * 43758.5453);
#     }

#     void main() {
#         vec4 textureFrag = texture2D(texture,gl_TexCoord[0].st);
#         gl_FragColor.rgb = textureFrag.rgb;
#     }
#     '''
# from psychopy import _shadersPyglet
# _shadersPyglet.fragFBOtoFrame = fragFBOtoFramePatched

# imports
# -------
from psychopy import core, visual, event, monitors
from settings import exp, db
from exputils import getFrameRate
from utils    import trim, to_percent
import numpy  as np
import yaml
import re

# setup monitor
monitorName = "testMonitor"
monitors = monitors.getAllMonitors()
if "BENQ-XL2411" in monitors:
	monitorName = "BENQ-XL2411"

# create a window
# ---------------
winkeys = {'units' : 'deg', 'fullscr' : True, 'useFBO' : True,
	'blendMode' : 'add', 'monitor' : monitorName}
if exp['two screens']:
	winkeys.update({'screen' : 0})	
# win = visual.Window(**winkeys)
win = visual.Window(monitor='testMonitor', fullscr=True, units='deg', 
    useFBO=True, blendMode='add')
win.setMouseVisible(False)


# STIMULI
# -------

def txt(win=win, **kwargs):
	return visual.TextStim(win, units='norm', **kwargs)


def txt_newlines(win=win, exp=exp, text='', **kwargs):
	text = text.replace('\\n', '\n')
	text = text.replace('[90button]', exp['keymap'][90])
	text = text.replace('[45button]', exp['keymap'][45])

	# check for gender related formulations
	ptrn = re.compile(r'\[(\w+)/(\w+)\]', flags=re.U)
	found = ptrn.finditer(text)
	new_text = ''; last_ind = 0
	for f in found:
		ind = f.span()
		grp = f.groups()
		correct_text = grp[exp['participant']['sex'] == 'k']
		new_text += text[last_ind:ind[0]] + correct_text
		last_ind = ind[1]
	if new_text:
		new_text += text[last_ind:]
		text = new_text

	return visual.TextStim(win, text=text, units='norm', **kwargs)


# gabor creation
def gabor(win = win, ori = 0, opa = 1.0,
		  pos  = [0, 0], size = exp['gabor size'],
		  units = 'deg', sf = exp['gabor freq']):
	return visual.GratingStim(win     = win,  mask = "gauss", \
							  size    = size, pos  = pos, \
							  sf      = sf,   ori  = ori,      \
							  opacity = opa,  units = units)


class Gabor(object):
	'''Simple gabor class that allows the contrast to go
	up to 3.0 (clipping the out of bound rgb values).
	Requires window to be set with useFBO=True and 
	blendMode='add' as well as a monkey-patch for
	pyglet shaders.'''

	def __init__(self, **kwargs):
		self.draw_which = 0
		win = kwargs.pop('win')
		self.contrast = kwargs.pop('contrast', 1.)
		kwargs.update({'contrast': 0.})

		# generate gabors:
		self.gabors = list()
		self.gabors.append(visual.GratingStim(win, **kwargs))
		self.gabors.append(visual.GratingStim(win, **kwargs))
		self.gabors.append(visual.GratingStim(win, **kwargs))
		self.set_contrast(self.contrast)

	def set_contrast(self, val):
		self.contrast = val
		self.draw_which = 0
		for g in self.gabors:
			if val > 0.:
				self.draw_which += 1
				if val > 1.:
					g.contrast = 1.
					val = val - 1.
				else:
					g.contrast = val
					break

	def draw(self):
		for g in range(self.draw_which):
			self.gabors[g].draw()


def fix(color=(0.5, 0.5, 0.5)):
	dot = visual.Circle(stim['window'], radius=0.15,
		edges=16, units='deg')
	dot.setFillColor(color)
	dot.setLineColor(color)
	return dot


def feedback_circle(win=win, radius=2.5, edges=64,
	color='green', pos=[0,0]):
	color_mapping = {'green': [0.1, 0.9, 0.1], 'red': [0.9, 0.1, 0.1]}
	color = color_mapping[color]
	circ = visual.Circle(win, pos=pos, radius=radius, edges=edges, units='deg')
	circ.setFillColor(color)
	circ.setLineColor(color)
	return circ


# prepare stimuli
# ---------------
stim = {}
stim['window'] = win

# resolve multiple screens stuff
if exp['two screens']:
	winkeys.update({'screen' : 1, 'blendMode' : 'avg'})
	stim['window2'] = visual.Window(**winkeys)
	imgwin = stim['window2']
else:
	imgwin = stim['window']

# create all target orientations
stim['target'] = dict()
for o in exp['orientation']:
	stim['target'][o] = Gabor(win=stim['window'],
		size = exp['gabor size'], units = 'deg',
		sf = exp['gabor freq'])

stim['centerImage'] = visual.ImageStim(imgwin, image=None,
	pos=(0.0, 0.0), size=(14*80,6*80), units = 'pix')

# mask - all gabor directions superposed
mask_ori = [0, 45, 90, 135]
stim['mask'] = []
for o in mask_ori:
	stim['mask'].append(gabor(ori = o, opa = 0.25))

# fixation and feedback circle:
stim['fix'] = fix()
stim['feedback'] = feedback_circle()


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

	# set target properties
	orientation = db.loc[tr]['orientation']
	contrast = db.loc[tr]['opacity']
	target = stim['target'][orientation]
	target.set_contrast(contrast)
	target_code = 'target_' + str(db.loc[tr]['orientation'])

	# get trial start time
	db.loc[tr, 'time'] = core.getTime()


	# PRESENT
	# -------

	# present fix:
	win.callOnFlip(onflip_work, exp['port'], code='fix')
	for f in np.arange(db.loc[tr]['fixTime']):
		stim['fix'].draw()
		win.flip()

	# clear keyboard buffer
	event.getKeys()

	# present target
	win.callOnFlip(onflip_work, exp['port'], code=target_code,
		clock=exp['clock'])
	for f in np.arange(db.loc[tr]['targetTime']):
		target.draw()
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


def present_training(exp=exp, slowdown=5, mintrials=10, corr=0.85):
	i = 1
	txt = u'Twoja poprawność:\n{}\n\ndocelowa poprawność:\n{}'
	txt += u'\n\n Aby przejść dalej naciśnij spację.'
	train_corr = 0
	train_db = give_training_db(db, slowdown=slowdown)

	while train_corr < corr or i < mintrials:
		present_trial(i, exp=exp, db=train_db)
		present_feedback(i, db=train_db)

		# check correctness
		train_corr = train_db.loc[max(1,
			i-mintrials+1):i, 'ifcorrect'].mean()

		if (i % mintrials) == 0 and train_corr < corr:
			# show info about correctness and remind key mapping
			thistxt = txt.format(to_percent(train_corr), to_percent(corr))
			textscreen(thistxt)
			show_resp_rules()

			# if this is the last training block - increase contrast
			if slowdown == 1:
				exp['opacity'] += 0.2
		i += 1
	# save training db!
	return train_db, train_corr


def present_feedback(i, db=db, stim=stim):
	if db.loc[i, 'ifcorrect'] == 1:
		stim['feedback'].setFillColor([0.1, 0.9, 0.1])
		stim['feedback'].setLineColor([0.1, 0.9, 0.1])
	else:
		stim['feedback'].setFillColor([0.9, 0.1, 0.1])
		stim['feedback'].setLineColor([0.9, 0.1, 0.1])

	for f in range(0, exp['fdb time'][0]):
		stim['feedback'].draw()
		stim['window'].flip()


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


class TimeShuffle(object):
	'''TimeShuffle is used to keep track of all fixation times
	that should be used in the experiment 2.

	parameters
	----------
	start - float; start value
	end   - float; end value
	every - float; step value
	times - int; number of times each value should be used

	example
	-------
	>>> times = TimeShuffle(start=1.5, end=3., every=0.5,
			times=2, shuffle=False).all()
	[ 1.5  2.   2.5  3.   1.5  2.   2.5  3. ]
	'''
	def __init__(self, start=1.5, end=5.0, every=0.05, times=6, shuffle=True):
		self.times = np.arange(start, end+0.001, every)
		self.inds = range(0, len(self.times)) * times
		if shuffle:
			np.random.shuffle(self.inds)
		self.current_ind = 0

	def get(self):
		givetime = self.times[self.inds[self.current_ind]]
		self.current_ind += 1
		return givetime

	def all(self):
		return self.times[self.inds]


# instructions etc.
# -----------------

# this should go to instructions module
def show_resp_rules(exp=exp, win=stim['window'], text=None):

	# create diagonal on one side and cardinal on the other
	ch    = int(exp['keymap'][45] == exp['use keys'][0])

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
	tx = [ u'naciśnij  ' + exp['keymap'][ornt[0]],
		   u'naciśnij  ' + exp['keymap'][ornt[2]],
		   u'aby przejść dalej naciśnij spację' ]
	positions = [[-0.5, 0.75], [0.5, 0.75], [0, -0.85]]
	txStim = []

	for t, p in zip(tx, positions):
		txStim.append(visual.TextStim(win, text = t,
			pos = p, units = 'norm'))

	# draw all:
	for t in txStim:
		t.draw()
	for g in stims:
		g.draw()

	# draw text if necessary:
	if text:
		visual.TextStim(win, text=text).draw()	

	win.flip()

	# wait for space:
	k = event.waitKeys(keyList = ['space'])


def textscreen(text, win=stim['window'], exp=exp):
	visual.TextStim(win, text = text, units = 'norm').draw()
	win.flip()
	event.waitKeys()


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
	mapdict    = {'gabor': gabor, 'text': txt_newlines,
		'fix':fix, 'feedback': feedback_circle}
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
		args = item['value']
		args.update({'win' : self.win})
		if fun: return fun(**item['value'])
