# -*- coding: utf-8 -*-

from psychopy       import visual, event, gui, core
from PIL            import Image
from utils          import round2step

import os
import re
import yaml
import numpy  as np
import pandas as pd


def plot_Feedback(stim, plotter, pth, keys=None, wait_time=5, resize=1.0):
	'''
	return feedback image in stim['centerImage']. Does not draw the image.
	'''
	# get file names from
	imfls = plotter.plot(pth)
	if not isinstance(imfls, type([])):
		imfls = [imfls]

	for im in imfls:
		# check image size:
		img = Image.open(im)
		imgsize = np.array(img.size)
		del img

		# clear buffer
		k = event.getKeys()

		# set image
		stim['centerImage'].size = np.round(imgsize * resize)
		stim['centerImage'].setImage(im)
		return stim


class ContrastInterface(object):

	def __init__(self, exp=None, stim=None):
		self.stim = stim
		self.exp  = exp
		self.contrast = []

		# monitor setup
		# -------------
		self.mouse = event.Mouse()
		self.two_windows = 'window2' in stim
		if self.two_windows:
			self.win = stim['window2']
			self.wait_txt = visual.TextStim(stim['window'],
				text=u'Proszę czekać, trwa dobieranie kontrastu...')
			self.wait_txt.draw()
			stim['window'].flip()
		else:
			self.win = stim['window']
		self.win.setMouseVisible(True)
		self.origunits = self.win.units
		self.win.units = 'norm'

		# postion ImageStim:
		self.stim['centerImage'].units = 'norm'
		self.stim['centerImage'].setPos((-0.4, 0.4))
		size = np.array(self.stim['centerImage'].size)
		print 'size pre: ', size
		if np.any(size > 1.5):
			prop = size[1]/size[0]
			self.stim['centerImage'].setSize((1.2, 1.2*prop))
			print 'size post: ', self.stim['centerImage'].size


		button_pos = np.zeros([4,2])
		button_pos[:,0] = 0.7
		button_pos[:,1] = np.linspace(-0.25, -0.8, num=4)
		button_text = [u'kontynuuj', u'zakończ', u'edytuj', '0.1']
		self.buttons = [Button(win=self.win, pos=p, text=t,
			size=(0.35, 0.12)) for p, t in zip(button_pos, button_text)]

		self.buttons[-1].click_fun = self.cycle_vals
		self.grain_vals = [0.1, 0.05, 0.01, 0.005]
		self.current_grain_val = 0
		self.text = visual.TextStim(win=self.win, text='kontrast:\n',
			pos=(0.75, 0.5), units='norm', height=0.1)
		self.text_height = {0: 0.12, 5: 0.1, 9: 0.06, 15: 0.04, 20: 0.03}

		# scale
		self.scale = ClickScale(win=self.win, pos=(0.,-0.8), size=(0.75, 0.1))
		self.edit_mode = False
		self.last_pressed = False


	def draw(self):
		[b.draw() for b in self.buttons]
		if self.buttons[-2].clicked:
			self.scale.draw()
			# TODO: this might be moved to refresh:
			if self.last_pressed:
				step = self.grain_vals[self.current_grain_val]
				self.contrast = round2step(np.array(self.scale.points), step=step)
				txt = 'kontrast:\n' + '\n'.join(map(str, self.contrast))
				num_cntrst = len(self.contrast)
				k = np.sort(self.text_height.keys())
				sel = np.where(np.array(k) <= num_cntrst)[0][-1]
				self.text.setHeight(self.text_height[k[sel]])
				self.text.setText(txt)
				self.last_pressed = False
			self.text.draw()
		self.stim['centerImage'].draw()


	def cycle_vals(self):
		self.current_grain_val += 1
		if self.current_grain_val >= len(self.grain_vals):
			self.current_grain_val = 0
		self.buttons[-1].setText(str(self.grain_vals[self.current_grain_val]))
		# TODO: change contrast values


	def refresh(self):
		if_click = self.check_mouse_click()
		self.last_pressed = if_click
		if not self.edit_mode and self.buttons[-1].clicked:
			self.edit_mode = True
		self.draw()
		self.win.flip(clearBuffer=False)
		if if_click:
			core.wait(0.1)


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
			
		elif m3:
			self.mouse.clickReset()
			self.scale.remove_point(-1)
		return m1 or m3


	def quit(self):
		self.win.setMouseVisible(False)
		self.win.units = self.origunits


class Button:
	'''
	Simple button class, does not check itself, needs to be checked
	in some event loop.

	create buttons
	--------------
	win = visual.Window(monitor="testMonitor")
	button_pos = np.zeros([3,2])
	button_pos[:,0] = 0.5
	button_pos[:,1] = [0.5, 0., -0.5]
	button_text = list('ABC')
	buttons = [Button(win=win, pos=p, text=t) for p, t in
		zip(button_pos, button_text)]

	draw buttons
	------------
	[b.draw() for b in buttons]
	win.flip()

	check if buttons were pressed
	-----------------------------
	mouse = event.Mouse()
	m1, m2, m3 = mouse.getPressed()
	if m1:
		mouse.clickReset()
		ifclicked = [b.contains(mouse) for b in buttons]
		which_clicked = np.where(ifclicked)[0]
		if which_clicked.size > 0:
			buttons[which_clicked[0]].click()
	'''

	def __init__(self, pos=(0, 0), win=None, size=(0.4, 0.15),
		text='...', box_color=(-0.3, -0.3, -0.3), font_height=0.08,
		units='norm', click_color=(0.2, -0.3, -0.3)):
		self.rect_stim = visual.Rect(win, pos=pos, width=size[0],
			height=size[1], fillColor=box_color, lineColor=box_color,
			units=units)
		self.text_stim = visual.TextStim(win, text=text, pos=pos,
			height=font_height, units=units)
		self.clicked = False
		self.orig_color = box_color
		self.click_color = click_color
		self.click_fun = self.default_click_fun


	def draw(self):
		self.rect_stim.draw()
		self.text_stim.draw()


	def contains(self, obj):
		return self.rect_stim.contains(obj)


	def setText(self, text):
		self.text_stim.setText(text)


	def default_click_fun(self):
		if self.clicked:
			self.clicked = False
			self.rect_stim.setFillColor(self.orig_color)
			self.rect_stim.setLineColor(self.orig_color)
		else:
			self.clicked = True
			self.rect_stim.setFillColor(self.click_color)
			self.rect_stim.setLineColor(self.click_color)


	def click(self):
		self.click_fun()


class ClickScale(object):

	def __init__(self, win=None, size=(0.5, 0.15), pos=(0, 0), 
		color=(-0.3, -0.3, -0.3), units='norm'):
		self.win = win
		# maybe - raise ValueError if win is none
		self.points = []
		self.lines = []
		self.units = units
		self.color = color
		self.pos = pos
		self.line_color = (1.0, -0.3, -0.3)
		self.scale = visual.Rect(win, pos=pos, width=size[0],
				height=size[1], fillColor=color, lineColor=color,
				units=units)
		# TODO: check if pos is divided by two
		self.x_extent = [pos[0]-size[0]/2., pos[0]+size[0]/2.]
		self.x_len = size[0]
		self.h = size[1]


	def point2xpos(self, point):
		return self.x_extent[0] + point*self.x_len


	def xpos2point(self, xpos):
		return (xpos - self.x_extent[0]) / self.x_len


	def test_click(self, mouse):
		if self.scale.contains(mouse):
			mouse_pos = mouse.getPos()
			val = self.xpos2point(mouse_pos[0])
			self.add_point(val)


	def add_point(self, val):
		self.points.append(val)
		pos = [0., self.pos[1]]
		pos[0] = self.point2xpos(val)
		ln = visual.Rect(self.win, pos=pos, width=0.01,
				height=self.h, fillColor=self.line_color, 
				lineColor=self.line_color, units=self.units)
		self.lines.append(ln)


	def remove_point(self, ind):
		try:
			self.points.pop(ind)
			self.lines.pop(ind)
		except:
			pass


	def draw(self):
		self.scale.draw()
		[l.draw() for l in self.lines]


def create_database(exp, trials=None, rep=None, combine_with=None):
	# define column names:
	colNames = ['time', 'fixTime', 'targetTime', 'SMI', \
				'maskTime', 'opacity', 'orientation', \
				'response', 'ifcorrect', 'RT']
	from_exp = ['targetTime', 'SMI', 'maskTime']

	# combined columns
	cmb = ['orientation']

	# how many times each combination should be presented
	if not trials:
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
		self.give_path(self, path_type, file_ending=file_ending)
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
