import os
import numpy  as np
import pandas as pd
from psychopy  import visual, core
from random    import randint, uniform #, choice
from exputils  import ms2frames, getUserName, continue_dataframe, \
					  getFrameRate

# experiment settings
# -------------------
exp = {}
exp['debug']       = True
exp['clock']       = core.Clock()
exp['use trigger'] = False
exp['port address'] = '0xDC00' # string, for example '0xD05'
exp['break after'] = 15 # how often subjects have a break

exp['participant'] = getUserName(intUser = False)

exp['targetTime']  = [1]
exp['SMI']         = [2] # Stimulus Mask Interval
exp['fixTimeLim']  = [0.75, 2.5]
exp['maskTime']    = [20]
exp['opacity']     = [0.05, 0.8]
exp['orientation'] = [0, 45, 90, 135]
exp['corrLims']    = [0.55, 0.9]

exp['use keys']    = ['f', 'j']
exp['respWait']    = 1.5

# response mapping
# ----------------
resp = {}
choose_resp = randint(0, 1)
resp[0]   = exp['use keys'][choose_resp]
resp[90]  = exp['use keys'][choose_resp]
resp[45]  = exp['use keys'][1 - choose_resp]
resp[135] = exp['use keys'][1 - choose_resp]
exp['keymap'] = resp
exp['choose_resp'] = choose_resp

# port settings
# -------------
portdict = {}
portdict['send'] = exp['use trigger']
portdict['port address'] = int(exp['port address'], base=16) \
						   if exp['port address'] and portdict['send'] \
						   else exp['port address']
portdict['codes'] = {'fix' : 1, 'mask' : 2}
portdict['codes'].update({'target_'+str(ori) : 4+i \
						   for i,ori in enumerate(exp['orientation'])
						   })
exp['port'] = portdict

# get path
pth   = os.path.dirname(os.path.abspath(__file__))
exp['path'] = pth

# ensure 'data' directory is available:
exp['data'] = os.path.join(pth, 'data')
if not os.path.isdir(exp['data']):
	os.mkdir(exp['data'])


# check frame rate:
win = visual.Window(monitor="testMonitor")
exp['frm'] = getFrameRate(win)
win.close()

# LOADING or CREATING EXPERIMENT DATAFRAME
# ----------------------------------------

# check if continue with previous dataframe:
ifcnt = continue_dataframe(exp['data'], exp['participant'] + '.xls')

if not ifcnt:
	# create DataFrame
	# ----------------
	# define column names:
	colNames = ['time', 'fixTime', 'targetTime', 'SMI', \
				'maskTime', 'opacity', 'orientation', \
				'response', 'ifcorrect', 'RT']
	# CHANGE - dtypes are not being used
	dtypes   = ['float', 'float', 'int', 'int', \
				'int', 'float', 'int', 'str', 'int', 'float']
	# combined columns
	cmb = ['targetTime', 'SMI', 'maskTime', 'orientation']
	# how many times each combination should be presented
	perComb = 140;
	# generate trial combinations
	lst = [
			[i, j, l, m ] \
				for i in exp['targetTime']  \
				for j in exp['SMI']         \
				for l in exp['maskTime']    \
				for m in exp['orientation'] \
			if i + j < 5] \
		* perComb

	# turn to array and shuffle rows
	lst = np.array(lst)
	np.random.shuffle(lst)

	# construct data frame
	shp = lst.shape
	db = pd.DataFrame(
		index = np.arange(1, shp[0] + 1),
		columns = colNames
		)
	exp['numTrials'] = len(db)

	# fill the data frame from lst
	for i, r in enumerate(cmb):
		db[r] = lst[:, i]

	# add fix time in frames
	db['fixTime'] = ms2frames( 
		np.random.uniform(
			low = exp['fixTimeLim'][0], 
			high = exp['fixTimeLim'][1], 
			size = exp['numTrials']
			) * 1000,
		exp['frm']['time']
		)

	startTrial = 1
else:
	db, startTrial = ifcnt
	exp['numTrials'] = len(db)
