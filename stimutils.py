# -*- coding: utf-8 -*-

# imports
# -------
import os
import re
import yaml
import time
from random import sample
import numpy  as np
import matplotlib.pyplot as plt

from psychopy import core, visual, event, monitors
from psychopy.monitors import Monitor

from settings import exp, db
from exputils import getFrameRate, trim_df
from utils    import trim, to_percent

if os.name == 'nt' and exp['use trigger']:
	from ctypes import windll


# setup monitor
monitor = "testMonitor"
if exp['lab monitor']:
	distance = exp['participant distance']
	monitor = Monitor('BenQ', width=53.136, distance=distance)
	monitor.setSizePix((1920, 1080))


# create a window
# ---------------
winkeys = {'units': 'deg', 'fullscr': True, 'useFBO': True, 'blendMode': 'add',
		   'monitor': monitor, 'size': (1920, 1080)}
if exp['two screens']:
	winkeys.update({'screen': 1})
win = visual.Window(**winkeys)
win.setMouseVisible(False)

stim = dict()
stim['window'] = win

# resolve multiple screens stuff
if exp['two screens']:
	winkeys.update({'screen' : 0, 'blendMode' : 'avg'})
	stim['window2'] = visual.Window(**winkeys)
	imgwin = stim['window2']
else:
	imgwin = stim['window']


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
def gabor(win=win, ori=0, opa=1.0, pos=(0, 0), size=exp['gabor size'],
		  units='deg', sf=exp['gabor freq']):
	return visual.GratingStim(win=win, mask="gauss", size=size, pos=pos,
							  sf=sf, ori=ori, opacity=opa, units=units)


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
		kwargs.update({'contrast': 0., 'mask': 'gauss'})

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
		# make sure window is in blendMode 'add':
		# if self.win.blendMode == 'add':
		# 	self.win.blendMode = 'add'
		for g in range(self.draw_which):
			self.gabors[g].draw()


def fix(win=win, color=(0.5, 0.5, 0.5)):
	dot = visual.Circle(win, radius=0.09, edges=16, units='deg',
						interpolate=True)
	dot.setFillColor(color)
	dot.setLineColor(color)
	return dot


def feedback_circle(win=win, radius=2.5, edges=64, color='green', pos=[0, 0]):
	color_mapping = {'green': [0.1, 0.9, 0.1], 'red': [0.9, 0.1, 0.1]}
	color = color_mapping[color]
	circ = visual.Circle(win, pos=pos, radius=radius, edges=edges,
						 units='deg', interpolate=True)
	circ.setFillColor(color)
	circ.setLineColor(color)
	return circ


# prepare stimuli
# ---------------

# create all target orientations
stim['target'] = dict()
for o in exp['orientation']:
	stim['target'][o] = Gabor(win=stim['window'],
		size=exp['gabor size'], units='deg',
		sf=exp['gabor freq'], ori=o)

stim['centerImage'] = visual.ImageStim(imgwin, image=None,
	pos=(0.0, 0.0), size=(14*80,6*80), units = 'pix',
	interpolate=True)

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
	if portdict['send']:
		windll.inpout32.Out32(portdict['port address'], 0)


# PRESENTATION
# ------------

def present_trial(trial, exp=exp, stim=stim, db=db, win=stim['window'],
				  contrast=1., monkey=None):
	# PREPARE
	# -------
	# randomize opacity if not set
	if contrast is not None:
		db.loc[trial, 'opacity'] = contrast
	else:
		contrast = db.loc[trial, 'opacity']

	# set target properties
	orientation = db.loc[trial, 'orientation']
	target = stim['target'][orientation]
	target.set_contrast(contrast)
	target_code = 'target_' + str(int(db.loc[trial, 'orientation']))

	# get trial start time
	db.loc[trial, 'time'] = core.getTime()


	# PRESENT
	# -------

	# present fix:
	win.callOnFlip(onflip_work, exp['port'], code='fix')
	for f in np.arange(db.loc[trial, 'fixTime']):
		stim['fix'].draw()
		win.flip()
		if f == 3:
			clear_port(exp['port'])

	# clear keyboard buffer, check for quit
	k = event.getKeys()
	if 'q' in k:
		core.quit()

	# present target
	win.callOnFlip(onflip_work, exp['port'], code=target_code,
				   clock=exp['clock'])
	for f in np.arange(db.loc[trial, 'targetTime']):
		target.draw()
		win.flip()

	# interval
	for f in np.arange(db.loc[trial, 'SMI']):
		win.flip()
	clear_port(exp['port'])

	# mask
	for f in np.arange(db.loc[trial, 'maskTime']):
		for m in stim['mask']:
			m.draw()
		win.flip()

	# response
	evaluate_response(db, exp, trial, monkey=monkey)

	# FIXME - use frames?
	# after 250 - 500 ms from response mask disappears
	offset = np.random.randint(25, 50) * 1.
	core.wait(offset / 100.)

	# send mask offset trigger
	win.callOnFlip(onflip_work, exp['port'], code='mask_offset')
	win.flip()
	core.wait(0.02)
	clear_port(exp['port'])


def evaluate_response(df, exp, trial, monkey=None):
	# which keys we wait for:
	keys = exp['use keys']
	if exp['debug']: keys += ['q']

	if monkey is None:
		# check if response
		k = event.getKeys(keyList=keys, timeStamped=exp['clock'])

		# wait for response (timeout)
		if not k: k = event.waitKeys(maxWait=exp['respWait'], keyList=keys,
									 timeStamped=exp['clock'])
	else:
		core.wait(0.1 + np.random.rand() * 0.2)
		k = [(monkey.respond(df.loc[trial, :]), 0.15)]

	# calculate RT and ifcorrect
	if k:
		key, RT = k[0]

		# send trigger
		onflip_work(exp['port'], code='response_{}'.format(key))
		core.wait(0.02)
		clear_port(exp['port'])

		# if debug - test for quit
		if exp['debug'] and key == 'q':
				core.quit()

		# performance
		df.loc[trial, 'response']  = key
		df.loc[trial, 'RT']        = RT
		target_ori = df.loc[trial]['orientation']
		df.loc[trial, 'ifcorrect'] = int(exp['keymap'][target_ori] == key)
	else:
		df.loc[trial, 'response']  = 'NoResp'
		df.loc[trial, 'ifcorrect'] = 0


# - [ ] TODO Monkey istead of auto
def present_training(trial, db, exp=exp, slowdown=5, mintrials=10, corr=0.8,
					 stim=stim, monkey=None, contrast=1., exp_info=None,
					 block_num=None):
	'''Present a block of training data.'''

	train_corr = 0
	auto = monkey is not None

	txt = u'Twoja poprawność:\n{}\n\ndocelowa poprawność:\n{}'
	txt += u'\n\n Aby przejść dalej naciśnij spację.'

	train_db = give_training_db(db, slowdown=slowdown)
	exp['opacity'] = np.array([1., 1.])
	exp['targetTime'] = [train_db.loc[trial, 'targetTime']]
	exp['SMI'] = [train_db.loc[trial, 'SMI']]

	if exp_info is not None:
		start_trial = trial - 1
		info_msg = (u'Trwa {} / {} blok treningowy.\nTrial {}\n'
					u'Obecna poprawność: {:.2f}')

	while train_corr < corr or trial < mintrials:
		stim['window'].flip()
		core.wait(0.5)
		present_trial(trial, exp=exp, db=db, contrast=contrast,
					  monkey=monkey)
		present_feedback(trial, db=db)

		# check correctness
		start_last = max(1, trial - mintrials + 1)
		train_corr = db.loc[start_last:trial, 'ifcorrect'].mean()

		if exp_info is not None:
			this_info_msg = info_msg.format(block_num[0], block_num[1],
											trial - start_trial, train_corr)
			exp_info.general_info(this_info_msg)

		if (trial % mintrials) == 0 and train_corr < corr:
			# show info about correctness and remind key mapping
			current_txt = txt.format(to_percent(train_corr), to_percent(corr))
			textscreen(current_txt, auto=auto)
			show_resp_rules(auto=auto)

			# FIX/CHECK - save opacity to db?
			if contrast < 3.:
				contrast += 0.5
		trial += 1
	# return db so it can be saved
	return train_db, train_corr, contrast


def present_feedback(i, db=db, stim=stim):
	if db.loc[i, 'ifcorrect'] == 1:
		stim['feedback'].setFillColor([0.1, 0.9, 0.1])
		stim['feedback'].setLineColor([0.1, 0.9, 0.1])
	else:
		stim['feedback'].setFillColor([0.9, 0.1, 0.1])
		stim['feedback'].setLineColor([0.9, 0.1, 0.1])

	stim['window'].blendMode = 'avg'
	for f in range(0, exp['fdb time'][0]):
		stim['feedback'].draw()
		stim['window'].flip()
	if 'window2' in stim:
		stim['window2'].blendMode = 'avg'
	stim['window'].blendMode = 'add'



# instructions etc.
# -----------------

# this should go to instructions module
def show_resp_rules(exp=exp, win=stim['window'], text=None, auto=False):

	# set up triggers for the break:
	win.callOnFlip(onflip_work, exp['port'], code='breakStart')

	# create diagonal on one side and cardinal on the other
	ch = int(exp['keymap'][45] == exp['use keys'][0])

	stims = []
	ornt = [[0, 90], [45, 135]]
	ornt = ornt[ch] + ornt[1 - ch]

	positions = [[-1, -0.43], [-1, 0.43], \
				 [1, -0.43],  [1, 0.43]]
	positions = np.array(positions) * 6.

	for o, p in zip(ornt, positions):
		stims.append(gabor(ori=o, pos=p, size=3., units='deg'))

	# add info on correct responses
	tx = [ u'naciśnij  ' + exp['keymap'][ornt[0]],
		   u'naciśnij  ' + exp['keymap'][ornt[2]],
		   u'aby przejść dalej naciśnij spację' ]
	positions = [[-1, 0.85], [1, 0.85], [0, -0.9]]
	positions = np.array(positions) * 6.
	txStim = []

	for t, p in zip(tx, positions):
		txStim.append(visual.TextStim(win, text=t, pos=p, units='deg',
									  height=0.7))

	# draw all:
	for t in txStim:
		t.draw()

	# draw text if necessary:
	if text:
		visual.TextStim(win, text=text, units='deg', height=0.7).draw()

	# fix window blendMode:
	win.blendMode = 'add'
	for g in stims:
		g.draw()
	win.flip()
	core.wait(0.02)
	clear_port(exp['port'])

	if not auto:
		# wait for space:
		k = event.waitKeys(keyList = ['space'])
	else:
		core.wait(0.1)

	# set end break trigger
	win.callOnFlip(onflip_work, exp['port'], code='breakStop')
	win.flip()
	core.wait(0.02)
	clear_port(exp['port'])


def textscreen(text, win=stim['window'], exp=exp, auto=False):
	visual.TextStim(win, text=text, units='norm').draw()
	# fix window blendMode:
	win.blendMode = 'add'
	win.flip()
	if not auto:
		event.waitKeys()
	else:
		core.wait(0.1)


def present_break(t, exp=exp, win=stim['window'], auto=False, correctness=None):
	tex  = u'Ukończono  {} / {}  powtórzeń.\nMożesz teraz ' + \
		   u'chwilę odetchnąc.\n{}Naciśnij spację aby kontynuowac...'
	add_text = ''
	if correctness is not None:
		add_text = u'Twoja poprawność wynosi: {:.1f}%\n'.format(correctness)
	tex  = tex.format(t, exp['numTrials'], add_text)
	info = visual.TextStim(win, text=tex, pos=[0, 0], units='norm')

	info.draw()
	# fix window blendMode:
	win.blendMode = 'add'
	win.flip()

	if not auto:
		# wait for space key:
		k = event.waitKeys(keyList=['space', 'escape', 'return'])
	else:
		core.wait(0.1)


def forced_break(win=stim['window'], auto=False, exp_info=None):
	tex  = u'Czas na obowiązkową przerwę,\npoczekaj na eksperymentatora.'
	info = visual.TextStim(win, text=tex, pos=[0, 0], units='norm')

	info.draw()

	# fix window blendMode:
	win.blendMode = 'add'
	win.flip()

	# update exp_info to alert experimenter
	exp_info.alert(text=u'Pora na\nchwilę\nprzerwy!')

	if not auto:
		# wait for space key:
		k = event.waitKeys(keyList=['x', 'q'])
	else:
		core.wait(10.)
	exp_info.general_info('procedura jedzie dalej!')


def final_info(corr, payout, win=stim['window'], auto=False, exp_info=None):
	tex  = (u'To już koniec badania,\ndziękujemy bardzo za wytrwały udział!\n'
			u'Twoja poprawność wyniosła: {:.1f}%\nWynagrodzenie: {} PLN')
	tex = tex.format(corr, payout)
	info = visual.TextStim(win, text=tex, pos=[0, 0], units='norm')

	info.draw()

	# fix window blendMode:
	win.blendMode = 'add'
	win.flip()

	# update exp_info to alert experimenter
	exp_info.alert(text=u'Koniec\nbadania!\nWynagrodzenie: {} PLN'.format(payout))

	if not auto:
		# wait for space key:
		k = event.waitKeys(keyList=['q', 'space'])
	else:
		core.wait(10.)


def break_checker(window, exp, df, exp_info, logfile, current_trial,
                  qp_refresh_rate=3, plot_fun=None, plot_arg=None, dpi=120,
                  img_name='temp.png', df_save_path='temp.xlsx',
                  show_completed=False, show_correctness=False,
                  use_forced_break=False):

    has_break = current_trial % exp['break after'] == 0
    has_forced_break = current_trial == 214 or current_trial == 428
    if has_break:
        save_df = trim_df(df)
        save_df.to_excel(df_save_path)

    if use_forced_break and has_forced_break:
        forced_break(win=window, auto=exp['debug'], exp_info=exp_info)

    if has_break or (has_forced_break and use_forced_break):
        # remind about the button press mappings
        show_resp_rules(exp=exp, auto=exp['debug'])
        if not show_completed: window.flip()

    if show_completed and (has_break or (has_forced_break and use_forced_break)):
        if show_correctness:
            upper_steps = df.query('step > 2')
            avg_corr = upper_steps.ifcorrect.mean() * 100
            present_break(current_trial, exp=exp, win=window, auto=exp['debug'],
                          correctness=avg_corr)
        else:
            present_break(current_trial, exp=exp, win=window, auto=exp['debug'])
        window.flip()

    if not exp['two screens']:
        # make sure feedback is only shown after the break on one screen
        qp_refresh_rate = 10000

    # visual feedback on parameters probability
    if current_trial % qp_refresh_rate == 0 or has_break:
        try:
            fig = plot_fun(plot_arg)
            fig.savefig(img_name, dpi=dpi)
            plt.close(fig)
            window.winHandle.activate()
        except:
            pass

        if not exp['two screens']:
            win.blendMode = 'avg'

        exp_info.experimenter_plot(img_name)

        if not exp['two screens']:
            event.waitKeys(['f', 'j', 'space', 'return'])
            win.blendMode = 'add'

        # quest plus refresh adds ~ 1 s to ITI so we prefer that
        # it is not predictable when refresh is going to happen
        return sample([3, 4, 5], 1)[0]
    return qp_refresh_rate


def give_training_db(db, exp=exp, slowdown=8):
	# copy the dataframe
	train_db = db.copy()
	# shuffle orientations
	orientations = train_db.loc[:, 'orientation'].values
	np.random.shuffle(orientations)
	train_db.loc[:, 'orientation'] = orientations

	# change targetTime, SMI (and maybe maskTime?)
	train_db.loc[:, 'targetTime'] = exp['targetTime'][0] * slowdown
	train_db.loc[:, 'SMI'] = exp['SMI'][0] * slowdown
	return train_db


class Instructions:
	mapdict = {'gabor': gabor, 'text': txt_newlines, 'fix': fix,
			   'feedback': feedback_circle}
	navigation = {'left': 'prev', 'right': 'next', 'space': 'next'}

	def __init__(self, fname, win=win, auto=False, exp_info=None,
				 images=False, image_args=dict()):
		self.win = win
		self.auto = auto
		self.images = images
		self.exp_info = exp_info

		# get instructions from file:
		if not images:
			with open(fname, 'r') as f:
			    instr = yaml.load_all(f)
			    self.pages = list(instr)
		else:
			self.pages = [visual.ImageStim(
				stim['window'], image=file, **image_args) for file in fname]
		self.this_page = 0
		self.stop_at_page = len(self.pages)

	def present(self, start=None, stop=None):
		if not isinstance(start, int):
			start = self.this_page
		if not isinstance(stop, int):
			stop = len(self.pages)

		# show pages:
		self.this_page = start
		while self.this_page < stop:
			# create page elements
			self.create_page()
			self.show_page()

			# wait for response
			if not self.auto:
				k = event.waitKeys(keyList=self.navigation.keys() + ['q'])[0]
				if 'q' in k:
					core.quit()
				action = self.navigation[k]
			else:
				core.wait(0.15)
				action = 'next'

			# go next/prev according to the response
			if action == 'next':
				self.this_page += 1
			else:
				self.this_page = max(0, self.this_page - 1)

	def create_page(self):
		if self.images: return
		self.pageitems = [self.parse_item(item)
						  for item in self.pages[self.this_page]]

	def show_page(self):
		if not self.images:
			# draw page elements
			any_circle = np.any([isinstance(x, visual.Circle)
				for x in self.pageitems])
			if any_circle:
				self.win.blendMode = 'avg'

			for it in self.pageitems:
				it.draw()
				if isinstance(it, visual.TextStim) and not any_circle:
					self.win.blendMode = 'add'
		else:
			self.pages[self.this_page].draw()
		self.win.flip()

		# inform experimenter about the progress
		if self.exp_info is not None:
			msg = 'Instrukcje, strona {}'.format(self.this_page)
			self.exp_info.general_info(msg)

	def parse_item(self, item):
		# currently: gabor or text:
		fun = self.mapdict.get(item['item'], None)
		args = item['value']
		args.update({'win' : self.win})
		if fun is not None:
			return fun(**item['value'])
