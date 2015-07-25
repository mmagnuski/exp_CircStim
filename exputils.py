# -*- coding: utf-8 -*-

from psychopy       import gui
from psychopy       import event
from PIL            import Image
import numpy  as np

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
