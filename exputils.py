# -*- coding: utf-8 -*-

from psychopy       import visual, event, gui
from PIL            import Image
import numpy  as np
import pandas as pd

# TODOs:
# [ ] - continue_dataframe should test for overwrite
#


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
		else:
			resp = event.waitKeys()
		return resp


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

	def draw(self):
		self.rect_stim.draw()
		self.text_stim.draw()

	def contains(self, obj):
		return self.rect_stim.contains(obj)

	def click(self):
		if self.clicked:
			self.clicked = False
			self.rect_stim.setFillColor(self.orig_color)
			self.rect_stim.setLineColor(self.orig_color)
		else:
			self.clicked = True
			self.rect_stim.setFillColor(self.click_color)
			self.rect_stim.setLineColor(self.click_color)


class ClickScale(object):
	# TODOs: add tickmarks - 0.25, 0.5, 0.75
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

	return db[1:trials]


def getFrameRate(win, frames = 25):
	# get frame rate
	print "testing frame rate..."
	frame = {}
	frame['rate'] = win.getActualFrameRate(nIdentical = frames)
	frame['time'] = 1000.0 / frame['rate']
	print "frame rate: " + str(frame['rate'])
	print "time per frame: " + str(frame['time'])
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
