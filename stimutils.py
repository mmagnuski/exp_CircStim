from psychopy import visual
from exputils import getFrameRate
import numpy  as np


# create a window
# ---------------
win = visual.Window([800,600],monitor="testMonitor", 
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
