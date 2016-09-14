# -*- coding: utf-8 -*-

import os
import re
import yaml

import numpy  as np
import pandas as pd

from psychopy import visual, event, gui, core
from PIL      import Image
from utils    import round2step
from gui import Button, ClickScale


def plot_Feedback(stim, plotter, pth, resize=1.0, plotter_args={},
				  set_size=True):
	'''
	return feedback image in stim['centerImage']. Does not draw the image.
	'''
	# get file names from
	if isinstance(plotter_args, dict) and plotter_args:
		imfls = plotter.plot(pth, **plotter_args)
	else:
		imfls = plotter.plot(pth)

	if not isinstance(imfls, type([])):
		imfls = [imfls]

	for im in imfls:
		# clear buffer
		k = event.getKeys()

		# set image
		stim['centerImage'].setImage(im)
		if set_size:
			# check image size:
			img = Image.open(im)
			imgsize = np.array(img.size)
			del img
			stim['centerImage'].size = np.round(imgsize * resize)
		return stim


class Interface(object):
	"""core Interface constructor - enables handling
	two screen displays"""
	exp = None
	stim = None
	def __init__(self, exp, stim, main_win=2):
		self.exp = exp
		self.stim = stim
		self.two_windows = 'window2' in stim
		if self.two_windows:
			main_w = 'window2' if main_win == 2 else 'window'
			sec_w  = 'window'  if main_win == 2 else 'window2'
			self.win = stim[main_w]
			self.win2 = stim[sec_w]

			if self.wait_text:
				self.wait_txt = visual.TextStim(stim[sec_w],
					text=self.wait_text)
				self.wait_txt.draw()
				stim[sec_w].flip()
		else:
			self.win = stim['window']


class ContrastInterface(Interface):

	def __init__(self, exp=None, stim=None, contrast_lims=(0., 3.),
		df=None, weibull=None):
		from weibull import fitw

		self.contrast = list()

		# monitor setup
		# -------------
		self.wait_text = u'Proszę czekać, trwa dobieranie kontrastu...'
		super(ContrastInterface, self).__init__(exp, stim)

		self.win.setMouseVisible(True)
		self.mouse = event.Mouse(win=self.win)

		# change units, size and position of centerImage:
		self.origunits = self.win.units
		self.win.units = 'norm'

		self.df = df.reset_index()
		self.use_lapse = False
		self.fitfun = fitw
		self.weibull = weibull
		self.num_trials = 40
		self.corr_steps = [0.55, 0.75, 0.9]

		win_pix_size = self.win.size
		pic_pix_size = self.stim['centerImage'].size
		pic_nrm_size = [(p / (w * 1.)) * 2. for (p, w) in
			zip(pic_pix_size, win_pix_size)]
		self.stim['centerImage'].units = 'norm'
		self.stim['centerImage'].setPos((-0.4, 0.4))
		self.stim['centerImage'].setSize([pic_nrm_size])

		# buttons
		button_pos = np.zeros([4,2])
		button_pos[:,0] = 0.7
		button_pos[:,1] = np.linspace(-0.25, -0.8, num=4)
		button_text = [u'kontynuuj', u'zakończ', u'edytuj', '0.1']
		self.buttons = [Button(win=self.win, pos=p, text=t,
			size=(0.35, 0.12)) for p, t in zip(button_pos, button_text)]
		self.buttons[-1].click_fun = self.cycle_vals
		self.last_trials = Button(win=self.win, pos=(-0.7, -0.43),
			text='{} trials'.format(self.num_trials), size=(0.35, 0.12))
		self.use_lapse_button = Button(win=self.win, pos=(-0.7, -0.6166),
			text='use lapse',size=(0.35, 0.12))
		self.last_trials.click_fun = self.set_last_trials
		self.use_lapse_button.click_fun = self.change_lapse

		# text
		self.text = visual.TextStim(win=self.win, text='kontrast:\n',
			pos=(0.75, 0.5), units='norm', height=0.1)
		self.text_height = {0: 0.12, 5: 0.1, 9: 0.06, 15: 0.04, 20: 0.03}
		self.scale_text = visual.TextStim(self.win, pos=(0.,-0.65), text="")
		self.contrast_levels_text = visual.TextStim(win=self.win, text='',
			pos=(0., -0.43))

		# scale
		self.scale = ClickScale(win=self.win, pos=(0.,-0.8), size=(0.75, 0.1),
								lims=contrast_lims)
		self.scale2 = ClickScale(win=self.win, pos=(0.,-0.9), size=(0.75, 0.05),
								 lims=contrast_lims)
		self.edit_mode = False
		self.last_pressed = False
		self.orig_contrast_lims = contrast_lims
		self.current_contrast_lims = contrast_lims
		self.grain_vals = [0.1, 0.05, 0.01, 0.005]
		self.current_grain_val = 0
		self.in_loop2 = False

		# back in how many trials:
		self.next_trials = 4
		self.trials_text = visual.TextStim(win=self.win,
			text=u'wróć po {} trialach'.format(self.next_trials),
			pos=(0., -0.45), units='norm', height=0.1)

	def draw(self):
		[b.draw() for b in self.buttons]
		self.last_trials.draw()
		self.use_lapse_button.draw()
		self.trials_text.draw()
		self.contrast_levels_text.draw()
		self.stim['centerImage'].draw()

		# edit constrast mode
		if self.buttons[-2].clicked:
			self.set_scale_text()
			self.scale.draw()
			self.scale2.draw()
			self.scale_text.draw()
			# TODO: this might be moved to refresh:
			if self.last_pressed:
				step = self.grain_vals[self.current_grain_val]
				self.contrast = round2step(np.array(self.scale.points),
										   step=step)
				txt = 'kontrast:\n' + '\n'.join(map(str, self.contrast))
				num_cntrst = len(self.contrast)
				k = np.sort(self.text_height.keys())
				sel = np.where(np.array(k) <= num_cntrst)[0][-1]
				self.text.setHeight(self.text_height[k[sel]])
				self.text.setText(txt)
				self.last_pressed = False
			self.text.draw()

	def set_last_trials(self):
		self.in_loop2 = True
		self.last_trials.default_click_fun()
		self.loop2()

	def change_lapse(self):
		self.use_lapse_button.default_click_fun()
		self.use_lapse = self.use_lapse_button.clicked
		self.refresh_weibull()

	def cycle_vals(self):
		self.current_grain_val += 1
		if self.current_grain_val >= len(self.grain_vals):
			self.current_grain_val = 0
		self.buttons[-1].setText(str(self.grain_vals[self.current_grain_val]))
		# TODO: change contrast values?

	def refresh(self):
		self.check_key_press()
		if_click = self.check_mouse_click()
		self.last_pressed = if_click
		if not self.edit_mode and self.buttons[-1].clicked:
			self.edit_mode = True
		self.draw()
		self.win.flip()
		if if_click:
			core.wait(0.1)

	def set_scale_text(self):
		for obj in [self.scale, self.scale2]:
			val = obj.test_pos(self.mouse)
			if val:
				step = self.grain_vals[self.current_grain_val]
				val = round2step(val, step=step)
				get_chars = 2 + np.sum(np.array([1., 0.1, 0.01, 0.001]) <= val)
				val = str(val)
				get_chars = min(get_chars, len(val))
				val = val[:get_chars]
				break
			else:
				val = ""
		self.scale_text.setText(val)

	def check_scale2(self):
		vals = sorted(self.scale2.points)
		if vals:
			step = self.grain_vals[self.current_grain_val]
			vals = round2step(np.asarray(vals), step=step)
			if len(vals) == 1:
				self.current_contrast_lims = (0., vals[0])
			elif len(vals) >= 2:
				self.current_contrast_lims = (vals[0], vals[-1])
		else:
			self.current_contrast_lims = self.orig_contrast_lims
		self.scale.set_lim(self.current_contrast_lims)

	def check_mouse_click(self):
		m1, m2, m3 = self.mouse.getPressed()
		if m1:
			self.mouse.clickReset()
			# test buttons
			ifclicked = [b.contains(self.mouse) for b in self.buttons]
			which_clicked = np.where(ifclicked)[0]
			if which_clicked.size > 0:
				self.buttons[which_clicked[0]].click()

			# test scale
			self.scale.test_click(self.mouse)
			if_clicked = self.scale2.test_click(self.mouse)
			if if_clicked:
				self.check_scale2()

			# test trial num edit
			if self.last_trials.contains(self.mouse):
				self.last_trials.click()

			if self.use_lapse_button.contains(self.mouse):
				self.use_lapse_button.click()

		elif m3:
			self.mouse.clickReset()
			self.scale.test_click(self.mouse, 'remove')
			if_clicked = self.scale2.test_click(self.mouse, 'remove')
			if if_clicked:
				self.check_scale2()
		return m1 or m3

	def check_key_press(self):
		k = event.getKeys()
		if k:
			if 'minus' in k:
				self.next_trials -= 1
				self.next_trials = max(3, self.next_trials)
			if 'equal' in k:
				self.next_trials += 1
			if 'q' in k or 'return' in k:
				self.runLoop = False
			self.trials_text.setText(u'wróć po {} trialach'.format(
				self.next_trials))

	def refresh_weibull(self):
		# fit weibull
		nrow = self.df.shape[0]
		look_back = min(nrow, self.num_trials)
		look_back = max(5, look_back)
		self.num_trials = look_back
		ind = np.r_[nrow-look_back:nrow]
		params = [1., 1., 0.] if self.use_lapse else [1., 1.]
		self.weibull = self.fitfun(self.df, ind, init_params=params)
		# self.params = self.weibull.params

		self.stim = plot_Feedback(self.stim, self.weibull,
			self.exp['data'], plotter_args=dict(mean_points=True),
			set_size=False)
		self.stim['centerImage'].draw()

		# update contrast for specified correctness
		# self.contrast_for_corr = self.weibull.get_threshold(self.corr_steps)
		# txt = ['{}% - {:05.3f}'.format(str(corr * 100).split('.')[0], cntr)
		# 	for corr, cntr in zip(self.corr_steps, self.contrast_for_corr)]
		# txt = '; '.join(txt)
		# self.contrast_levels_text.setText(txt)

	def test_keys_loop2(self, k):
		if k and self.in_loop2:
			k = k[0]
			current_str = str(self.num_trials)
			if current_str == '0':
				current_str = ''
			# backspace - remove char from num_trials
			if k == 'backspace':
				if len(current_str) > 0:
					current_str = current_str[:-1]

			# number - add to num_trials
			if k in list('1234567890'):
				current_str += k

			self.num_trials = int(current_str) if \
				len(current_str) > 0 else 0
			txt = current_str if self.num_trials > 0 else ''
			self.last_trials.setText(current_str + ' trials')

			# return - refresh weibull
			if k == 'return':
				self.refresh_weibull()
				self.in_loop2 = False
				self.last_trials.default_click_fun()

	def loop2(self):
		while self.in_loop2:
			k = event.getKeys(
				keyList=list('1234567890') + ['return', 'backspace'])
			self.test_keys_loop2(k)
			self.draw()
			self.win.flip()

	def quit(self):
		self.win.setMouseVisible(False)
		self.win.units = self.origunits

	def loop(self):
		self.runLoop = True
		continue_fitting = True
		while self.runLoop:
			self.refresh()
			if self.buttons[1].clicked:
				self.runLoop = False
				continue_fitting = False
			elif self.buttons[0].clicked:
				self.runLoop = False
				continue_fitting = True
		self.quit()
		return continue_fitting


class ExperimenterInfo(Interface):
	def __init__(self, exp, stim):
		self.wait_text = False
		super(ExperimenterInfo, self).__init__(exp, stim)

		self.main_text = visual.TextStim(self.win, pos=(0, 0.5), units='norm')
		self.sub_text  = visual.TextStim(self.win, pos=(0, 0.25), units='norm')
		self.detail_text = visual.TextStim(self.win, pos=(0, 0), units='norm')
		self.textObjs = [self.main_text, self.sub_text, self.detail_text]

	def refresh(self):
		self.main_text.draw()
		self.sub_text.draw()
		self.win.flip()

	def update_text(self, texts):
		if self.two_windows and texts:
			update = False
			while len(texts) < len(self.textObjs):
				texts.append(None)
			for t, o in zip(texts, self.textObjs):
				if t:
					update = True
					o.setText(t)
			if update:
				self.refresh()

	def training_info(self, blockinfo, corr):
		text1 = u'Ukończono blok {0} \\ {1} treningu.'.format(*blockinfo)
		text2 = u'Uzyskano poprawność: {}'.format(corr)
		self.update_text([text1, text2])

	def blok_info(self, name, blockinfo):
		tx = list()
		tx.append(u'Trwa {}'.format(name))
		tx.append(u'Ukończono {} \ {} powtórzeń'.format(*blockinfo[:2]))
		self.update_text(tx)


class AnyQuestionsGUI(Interface):
	def __init__(self, exp, stim):
		self.wait_text = False
		super(AnyQuestionsGUI, self).__init__(exp, stim, main_win=1)
		tx = (u'Jeżeli masz jakieś pytania - naciśnij f.\n' +
			u'Jeżeli nie masz pytań - naciśnij spację.')
		self.tx1 = visual.TextStim(self.win, text=tx)
		if self.two_windows:
			self.tx2 = visual.TextStim(self.win2, text='...')
			self.circ = visual.Circle(self.win2, radius=6, edges=64)
			self.circ.setLineColor([0.9, 0.1, 0.1])
			self.circ.setFillColor([0.9, 0.1, 0.1])

	def run(self):
		if self.two_windows:
			self.tx2.draw()
			self.win2.flip()
		self.tx1.draw()
		self.win.flip()
		self.pressed = event.waitKeys(keyList=['space', 'f'])

		if not ('space' in self.pressed):
			if self.two_windows:
				self.tx2.setText(u'!!!!!!!!')
				self.circ.draw()
				self.tx2.draw()
				self.win2.flip()
			self.tx1.setText(u'Poczekaj na eksperymentatora.')
			self.tx1.draw()
			self.win.flip()
			self.pressed = event.waitKeys(keyList=['i', 'o'])


class FinalFitGUI(Interface):

	def __init__(self, exp=None, stim=None, db=None, weibull=None,
		fitfun=None, corr_steps=None, num_trials=40, use_lapse=None):

		# setup
		# -----
		self.wait_text = u'Proszę czekać, trwa dobieranie kontrastu...'
		super(FinalFitGUI, self).__init__(exp, stim)

		self.win.setMouseVisible(True)
		self.mouse = event.Mouse(win=self.win)
		self.db = db

		if weibull is not None:
			self.weibull = weibull
			self.params = weibull.params
			if use_lapse is not None:
				correct_num_params = 3 if use_lapse else 2
				assert len(self.params) == correct_num_params
			else:
				use_lapse = len(self.params) == 3
		else:
			self.weibull = []
			self.params = []

		# weibull-related options:
		self.img_size = []
		self.num_trials = num_trials
		self.corr_steps = [0.55, 0.75, 0.95] if corr_steps is None \
												else corr_steps
		self.contrast_for_corr = []
		self.fitfun = fitfun
		self.use_lapse = False if use_lapse is None else use_lapse

		# OK button
		pos = [0., -0.85]
		txt = 'OK'
		self.OKbutton = Button(win=self.win, pos=pos, text=txt,
							   size=(0.35, 0.12))
		self.OKbutton.click_fun = self.accept
		self.lapse_button = Button(win=self.win, pos=(0.6, -0.85), text='use lapse',
								   size=(0.35, 0.12))
		self.lapse_button.click_fun = self.change_lapse
		self.notfinished = True

		# edit box
		self.text = visual.TextStim(win=self.win, text=str(self.num_trials),
									pos=(0.0, -0.5), units='norm', height=0.1)
		# contrast for % corr:
		self.text2 = visual.TextStim(
			win=self.win, text='Contrast for correctness: ',
			pos=(0.0, -0.65), units='norm', height=0.08)

		# refresh weibull
		self.refresh_weibull()

		# centerImage (weibull fit plot):
		self.origunits = self.win.units
		pic = self.stim['centerImage']
		ypos = scale_img(self.win, pic, (0.1, -0.45))
		self.img_size = list(pic.size)

		self.win.units = 'norm'
		pic.setPos((0., ypos))

	def draw(self):
		self.OKbutton.draw()
		self.lapse_button.draw()
		self.text.draw()
		self.text2.draw()
		self.stim['centerImage'].draw()

	def change_lapse(self):
		self.use_lapse = False if self.use_lapse else True
		self.lapse_button.default_click_fun()
		self.refresh_weibull()

	def refresh_weibull(self):
		# fit weibull
		nrow = self.db.shape[0]
		look_back = min(nrow, self.num_trials)
		look_back = max(5, look_back)
		ind = np.r_[nrow-look_back:nrow]
		params = [1., 1., 0.] if self.use_lapse else [1., 1.]
		self.weibull = self.fitfun(self.db, ind, init_params=params)
		self.params = self.weibull.params

		self.stim = plot_Feedback(self.stim, self.weibull,
			self.exp['data'], plotter_args=dict(mean_points=True))
		if self.img_size:
			self.stim['centerImage'].size = self.img_size
		self.stim['centerImage'].draw()

		# update contrast for specified correctness
		self.contrast_for_corr = self.weibull.get_threshold(self.corr_steps)
		txt = ['{}% - {:05.3f}'.format(str(corr * 100).split('.')[0], cntr)
			for corr, cntr in zip(self.corr_steps, self.contrast_for_corr)]
		txt = '; '.join(txt)
		self.text2.setText(txt)

	def test_keys(self, k):
		if k:
			k = k[0]
			current_str = str(self.num_trials)
			if current_str == '0':
				current_str = ''
			# backspace - remove char from num_trials
			if k == 'backspace':
				if len(current_str) > 0:
					current_str = current_str[:-1]

			# number - add to num_trials
			if k in list('1234567890'):
				current_str += k

			self.num_trials = int(current_str) if \
				len(current_str) > 0 else 0
			txt = current_str if self.num_trials > 0 else ''
			self.text.setText(current_str)

			# return - refresh weibull
			if k == 'return':
				self.refresh_weibull()

	def loop(self):
		"""runs main GUI loop"""

		while self.notfinished:
			k = event.getKeys(
				keyList=list('1234567890') + ['return', 'backspace'])
			self.test_keys(k)
			self.check_mouse_click()

			self.draw()
			self.win.flip()

	def check_mouse_click(self):
		m1, m2, m3 = self.mouse.getPressed()
		if m1:
			self.mouse.clickReset()
			# test buttons
			if self.OKbutton.contains(self.mouse):
				self.OKbutton.click()
			if self.lapse_button.contains(self.mouse):
				self.lapse_button.click()

	def accept(self):
		self.notfinished = False


def scale_img(win, img, y_margin):
	winsize = win.size
	imsize = img.size
	imh_norm = imsize[1] / winsize[1]
	ypos = 1. - y_margin[0] - imh_norm
	low_margin = ypos - imh_norm
	ypos = int(ypos * winsize[1] / 2.)

	if low_margin < y_margin[1]:
		too_long_prop = (y_margin[1] - low_margin) / imh_norm
		img.size = np.array(imsize) * (1-too_long_prop)
		ypos = scale_img(win, img, y_margin)

	return ypos


def create_database(exp, trials=None, rep=None, combine_with=None):
	# define column names:
	colNames = ['time', 'fixTime', 'targetTime', 'SMI', \
				'maskTime', 'opacity', 'orientation', \
				'response', 'ifcorrect', 'RT']
	from_exp = ['targetTime', 'SMI', 'maskTime']

	# combined columns
	cmb = ['orientation']

	# how many times each combination should be presented
	if not trials and not rep:
		trials = 140

	# generate trial combinations
	if not combine_with:
		lst = exp['orientation']
		num_rep = rep if not rep == None else np.ceil(trials / float(len(lst)))
		lst = np.reshape(np.array(lst * num_rep), (-1, 1))
	else:
		cmb.append(combine_with[0])
		lst = [[o, x] for o in exp['orientation'] for x in combine_with[1]]
		num_rep = rep if not rep == None else np.ceil(trials / float(len(lst)))
		lst = np.array(lst * num_rep)

	# turn to array and shuffle rows
	np.random.shuffle(lst)

	# construct data frame
	db = pd.DataFrame(
		index = np.arange(1, lst.shape[0] + 1),
		columns = colNames
		)

	# exp['numTrials'] = len(db)
	num_trials = db.shape[0]
	if not trials:
		trials = num_trials

	# fill the data frame from lst
	for i, r in enumerate(cmb):
		db[r] = lst[:, i]

	# add fix time in frames
	if 'fixTime' not in cmb:
		db['fixTime'] = ms2frames(
			np.random.uniform(
				low = exp['fixTimeLim'][0],
				high = exp['fixTimeLim'][1],
				size = num_trials
				) * 1000,
			exp['frm']['time']
			)

	# fill relevant columns from exp:
	for col in from_exp:
		db[col] = exp[col][0]

	# fill NaNs with zeros:
	db.fillna(0, inplace=True)

	# make sure types are correct:
	to_float = ['time', 'opacity', 'RT']
	to_int   = ['ifcorrect', 'fixTime', 'targetTime', 'SMI', 'maskTime']
	db.loc[:, to_float+to_int] = db.loc[:, to_float+to_int].fillna(0)
	db.loc[:, to_float] = db.loc[:, to_float].astype('float64')
	db.loc[:, to_int] = db.loc[:, to_int].astype('int32')

	return db[0:trials]


def getFrameRate(win, frames=25):
	# get frame rate
	frame = {}
	frame['rate'] = win.getActualFrameRate(nIdentical = frames)
	frame['time'] = 1000.0 / frame['rate']
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


# get user name:
def getSubject():
    '''
    PsychoPy's simple GUI for entering 'stuff'
    Does not look nice or is not particularily user-friendly
    but is simple to call.
    Here it just asks for subject's name/code
    and returns it as a string
    '''
    myDlg = gui.Dlg(title="Pseudonim", size = (800,600))
    myDlg.addText('Informacje o osobie badanej')
    myDlg.addField('numer:')
    myDlg.addField('wiek:', 30)
    myDlg.addField(u'płeć:', choices=[u'kobieta', u'mężczyzna'])
    myDlg.show()  # show dialog and wait for OK or Cancel

    if myDlg.OK:  # the user pressed OK
        user = myDlg.data
    else:
        user = None

    return user


class DataManager(object):
	'''manges data paths for the experiment - avoiding overwrites etc.'''
	def __init__(self, exp):
		self.keymap = exp['keymap']
		self.choose_resp = exp['choose_resp']
		self.ID = exp['participant']['ID']
		self.age = exp['participant']['age']
		self.sex = exp['participant']['sex']
		self.val = dict()
		self.path = dict()
		self.path['data'] = exp['data']
		self.path['ID'] = os.path.join(self.path['data'], self.ID)

		# check if such subject has been created
		# if so - read keymap
		self.subj_present = os.path.isfile(self.path['ID'])
		if self.subj_present:
			self.read()
		else:
			self.write()


	def read(self):
		with open(self.path['ID'], 'r') as f:
			data = yaml.load(f)
		self.keymap = data['key-mapping']
		self.age = data['age']
		self.sex = data['sex']


	def give_path(self, path_type, file_ending='xls'):
		if path_type in self.path and self.path[path_type]:
			return self.path[path_type]
		else:
			pattern = self.ID + '_{}_([0-9]+)\.'.format(path_type) + file_ending
			def get_repval(pattern, s):
				r = re.search(pattern, s)
				return int(r.groups()[0]) if r else 0
			fls = os.listdir(self.path['data'])
			self.val[path_type] = max([get_repval(pattern, s) for s in fls]) + 1
			self.path[path_type] = os.path.join(self.path['data'], self.ID +
				'_{}_{}.'.format(path_type, self.val[path_type]) + file_ending)
			return self.path[path_type]


	def give_previous_path(self, path_type, file_ending='xls'):
		# make sure current path was checked
		self.give_path(path_type, file_ending=file_ending)
		# get previous:
		prev_val = self.val[path_type] - 1
		if prev_val == 0:
			return ''
		else:
			return os.path.join(self.path['data'], self.ID + '_{}_{}.'.format(
				path_type, prev_val) + file_ending)


	def write(self):
		save_data = {'ID': self.ID, 'age': self.age,
			'sex':self.sex, 'key-mapping': self.keymap}
		with open(self.path['ID'], 'w') as f:
			f.write(yaml.dump(save_data))


	def update_exp(self, exp):
		exp['choose_resp'] = self.choose_resp
		exp['keymap'] = self.keymap
		exp['participant']['sex'] = self.sex
		exp['participant']['age'] = self.age
		return exp
