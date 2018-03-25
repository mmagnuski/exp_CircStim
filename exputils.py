# -*- coding: utf-8 -*-

import os
import re
import yaml

import numpy  as np
import pandas as pd

from psychopy import visual, event, gui, core
from PIL      import Image
from utils    import round2step, trim_df
from gui import Button, ClickScale, Interface


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

	if not isinstance(imfls, list):
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



class ExperimenterInfo(Interface):
	def __init__(self, exp, stim, main_text_pos=(0, 0.5),
				 sub_text_pos=(0, 0.25)):
		self.wait_text = False
		super(ExperimenterInfo, self).__init__(exp, stim)

		# texts
		main_text = visual.TextStim(self.win, pos=main_text_pos, units='norm')
		sub_text  = visual.TextStim(self.win, pos=sub_text_pos, units='norm')
		detail_text = visual.TextStim(self.win, pos=(0, 0), units='norm')
		self.texts = dict(main=main_text, sub=sub_text, detail=detail_text)

		# center image
		self.image = stim['centerImage'] if 'centerImage' in stim else None

	def refresh(self):
		for txt in self.texts.values():
			txt.draw()

		if self.image is not None:
			self.image.draw()
		self.win.flip()

	def update_text(self, texts):
		update = False
		if not (self.two_windows and texts):
			return None

		# pad texts with None
		while len(texts) < len(self.texts):
			texts.append(None)

		for txt, key in zip(texts, ['main', 'sub', 'detail']):
			if txt is not None:
				update = True
				self.texts[key].setText(txt)

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

	def general_info(self, text):
		texts = [None, None, text]
		self.update_text(texts)

	def experimenter_plot(self, img_name):
		self.image.setImage(img_name)
		img_size = np.array(Image.open(img_name).size)
		self.image.size = img_size # np.round(imgsize * resize)
		self.refresh()


class AnyQuestionsGUI(Interface):
	def __init__(self, exp, stim, auto=False):
		self.wait_text = False
		super(AnyQuestionsGUI, self).__init__(exp, stim, main_win=1)
		tx = (u'Jeżeli coś nie jest jasne / masz jakieś pytania - naciśnij '
			  u'f.\nJeżeli wszystko jest jasne i nie masz pytań - naciśnij '
			  u'spację.')
		self.tx1 = visual.TextStim(self.win, text=tx)
		self.auto = auto
		if self.two_windows:
			self.tx2 = visual.TextStim(self.win2, text='...', height=5.)
			self.circ = visual.Circle(self.win2, radius=6, edges=64)
			self.circ.setLineColor([0.9, 0.1, 0.1])
			self.circ.setFillColor([0.9, 0.1, 0.1])

	def run(self):
		if self.two_windows:
			self.tx2.draw()
			self.win2.flip()
		self.tx1.draw()
		self.win.flip()

		if not self.auto:
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
				self.pressed = event.waitKeys(keyList=['q', 'return'])
				if self.two_windows:
					self.win2.flip()
		else:
			core.wait(0.15)


# TODO
# - [ ] add some of functionality below to chainsaw
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

	# generate all trial combinations
	if not combine_with:
		lst = exp['orientation']
		num_rep = (int(rep) if not rep is None else
			int(np.ceil(trials / float(len(lst)))))
		lst = np.reshape(np.array(lst * num_rep), (-1, 1))
	else:
		cmb.append(combine_with[0])
		lst = [[o, x] for o in exp['orientation'] for x in combine_with[1]]
		num_rep = rep if not rep is None else np.ceil(trials / float(len(lst)))
		lst = np.array(lst * num_rep)

	# turn to array and shuffle rows, then construct dataframe
	np.random.shuffle(lst)
	db = pd.DataFrame(index=np.arange(1, lst.shape[0] + 1), columns=colNames)

	# exp['numTrials'] = len(db)
	num_trials = db.shape[0]
	if not trials:
		trials = num_trials

	# fill the data frame from lst
	for i, r in enumerate(cmb):
		db[r] = lst[:, i]

	# add fix time in frames
	if 'fixTime' not in cmb:
		time_limits = exp['fixTimeLim']
		fix_ms = np.random.uniform(low=time_limits[0], high=time_limits[1],
								   size=num_trials) * 1000
		db['fixTime'] = ms2frames(fix_ms, exp['frm']['time'])

	# fill relevant columns from exp:
	for col in from_exp:
		db[col] = exp[col][0]

	# fill NaNs with zeros:
	db.fillna(0, inplace=True)

	# make sure types are correct:
	to_float = ['time', 'opacity', 'RT']
	to_int = ['ifcorrect', 'fixTime', 'targetTime', 'SMI', 'maskTime']
	db.loc[:, to_float + to_int] = db.loc[:, to_float + to_int].fillna(0)
	db.loc[:, to_float] = db.loc[:, to_float].astype('float64')
	db.loc[:, to_int] = db.loc[:, to_int].astype('int32')

	return db[0:trials]


def getFrameRate(win, frames=25):
	frame_rate = win.getActualFrameRate(nIdentical=frames)
	return dict(rate=frame_rate, time=1000.0 / frame_rate)


def ms2frames(times, frame_time):
	tp = type(times)
	if isinstance(times, list):
	    frms = list()
	    for t in times:
	        frms.append(int(round(t / frame_time)))
	elif isinstance(times, dict):
	    frms = dict()
	    for t in times.keys():
	        frms[t] = int(round(times[t] / frame_time))
	elif isinstance(times, np.ndarray):
		frms = np.array(np.round(times / frame_time), dtype=int)
	else:
		raise ValueError('times has to be list, dict or numpy ndarray.')
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
    myDlg = gui.Dlg(title="GabCon 2018", size = (800,600))
    myDlg.addText('Informacje o osobie badanej')
    myDlg.addField('numer:')
    myDlg.addField('plec:', choices=['kobieta', 'mezczyzna'])
    myDlg.addField('distance (cm)', 95.)
    myDlg.addText('Ustawienia procedury')
    myDlg.addField('debug mode:', choices=['False', 'True'])
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
		save_data = {'ID': self.ID, 'sex': self.sex,
					 'key-mapping': self.keymap}
		with open(self.path['ID'], 'w') as f:
			f.write(yaml.dump(save_data))

	def update_exp(self, exp):
		exp['choose_resp'] = self.choose_resp
		exp['keymap'] = self.keymap
		exp['participant']['sex'] = self.sex
		return exp


def get_valid_path(paths):
	for pth in paths:
		if os.path.exists(pth):
			return pth
	raise ValueError('could not find valid path.')
