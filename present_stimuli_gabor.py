#
# shebang with env\python etc. ? 

# this is a simple test for first grant experiment
# (gabor brightness)

# TODOs:
# [ ] add fixation cross
# [ ] add keypresses to database
# [ ] add simple instructions
# [ ] add simple training

# imports
print 'importing psychopy...'
from psychopy import visual, core, event
from exputils import getFrameRate
from random   import randint, uniform, choice
import numpy  as np
import pandas as pd

# experiment settings
exp = {}
exp['debug']       = True
exp['use trigger'] = False
exp['break after'] = 10
exp['targetTime']  = [1, 2, 3]
exp['SMI']         = [1, 2, 3] # Stimulus Mask Interval
exp['maskTime']    = [20]
exp['opacity']     = [0.05, 0.1, 0.2, 0.4]
exp['orientation'] = [0, 45, 90, 135]

exp['use keys']    = ['f', 'j']
exp['respWait']    = 1.5

resp = {}
choose_resp = randint(0, 1)
resp[0]   = exp['use keys'][choose_resp]
resp[90]  = exp['use keys'][choose_resp]
resp[45]  = exp['use keys'][1 - choose_resp]
resp[135] = exp['use keys'][1 - choose_resp]
exp['keymap'] = resp
exp['choose_resp'] = choose_resp



# define column names:
colNames = ['targetTime', 'SMI', 'maskTime', 'opacity', \
			'orientation', 'response', 'ifcorrect', 'RT']

# generate trials
cmb = ['targetTime', 'SMI', 'opacity', 'maskTime', 'orientation']
dtp = ['']
perComb = 3;
lst = [[i, j, k, l,m ] for i in exp['targetTime'] \
					   for j in exp['SMI'] \
					   for k in exp['opacity'] \
					   for l in exp['maskTime'] \
					   for m in exp['orientation']] \
					   * perComb

# turn to list and shuffle
lst = np.array(lst)
np.random.shuffle(lst)

# construct data frame
shp = lst.shape
db = pd.DataFrame(index = np.arange(1, shp[0] + 1), \
				  columns = colNames )

for i, r in enumerate(cmb):
	db[r] = lst[:, i]

print "total number of trials:", len(db)


# create a window
win = visual.Window([800,600],monitor="testMonitor", 
    units="deg", color = [-0.2, -0.2, -0.2], fullscr=False)
win.mouseVisible = False

# get fame rate
# frm = getFrameRate(win)

# ==def for gabor creation==
def gabor(win = win, ori = 0, opa = 1.0, pos  = [0, 0], size = 7, units = 'deg', sf = 1.5):
	return visual.GratingStim(win     = win,  mask = "gauss", \
							  size    = size, pos  = pos, \
							  sf      = sf,   ori  = ori,      \
							  opacity = opa,  units = units)

# prepare stimuli
stim = {}
stim['target'] = gabor()

mask_ori = [0, 45, 90, 135]
stim['mask'] = []

# mask is more complex story: 
# this could be either noise (?)
# or superposed gabor directions
# currently we use superposed gabors

for o in mask_ori:
	stim['mask'].append(gabor(ori = o, opa = 0.25))

# create a separate procedure for presenting 
# stimuli when first frame sends trigger and
# records time

# target
tr = 1
stim['target'].ori = db.iloc[tr]['orientation']
stim['target'].opacity = db.iloc[tr]['opacity']

# get time (should be taken on first frame)
t1 = core.getTime()
for f in np.arange(120):
	stim['target'].draw()
	win.flip()

# interval
for f in np.arange(120):
	win.flip()

# mask
for f in np.arange(120):
	for m in stim['mask']:
		m.draw()
	win.flip()