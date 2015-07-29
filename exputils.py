# -*- coding: utf-8 -*-

from psychopy       import gui
from psychopy       import event
from PIL            import Image
import numpy  as np
import pandas as pd

# TODOs:
# [ ] - continue_dataframe should test for overwrite
#


def plot_Feedback(stim, plotter, pth, keys=None, wait_time=5, resize=1.0):
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
		stim['centerImage'].draw()
		stim['window'].update()

		if keys:
			resp = event.waitKeys(keyList=keys, maxWait=wait_time)
		else:
			resp = event.waitKeys()
		return resp


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
