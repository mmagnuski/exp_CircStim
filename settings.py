# imports
import os
import numpy  as np
import pandas as pd
from psychopy  import visual, core
from random    import randint, uniform
from utils     import continue_dataframe
from exputils  import (ms2frames, getSubject,
	getFrameRate, create_database)

# experiment settings
# -------------------
exp = dict()
exp['debug']        = True
exp['two screens']  = True
exp['use trigger']  = True

exp['run blinkdot'] = False
exp['run baseline1'] = False
exp['run instruct'] = True
exp['run training'] = True
exp['run fitting']  = True
exp['run main c']   = True
exp['run main t']   = True
exp['break after']  = 15  # how often subjects have a break
exp['run baseline2'] = False

exp['port address'] = '0xDC00' # string, for example '0xD05'
exp['clock']        = core.Clock()

# contrast settings
exp['corrLims']    = [0.55, 0.95]
exp['opacity']     = [0.05, 0.8]
exp['min opac']    = 0.005
exp['max opac']    = 2.0
exp['opac steps']  = 10

# training settings
exp['train slow']   = [8, 5, 3, 2, 1]
exp['train corr']   = [0.8, 0.8, 0.8, 0.9, 0.9]

# fitting settings
exp['step until']   = [10, 35]  # continue stepwise until this trial
exp['fit until']    = 75 # continue fitting until this trial
exp['fitCorrLims']  = [0.55, 0.95]
exp['search method']= '4steps'
exp['max fit']      = 200

# timing settings
exp['targetTime']  = [2] # maybe test with targetTime == 2?
exp['SMI']         = [1] # Stimulus Mask Interval
exp['fixTimeLim']  = [1., 2.5]
exp['maskTime']    = [20]
exp['fdb time']    = [40] # feedback time in frames
exp['respWait']    = 1.5

# gabor settings
exp['gabor size']  = 5
exp['gabor freq']  = 1.5
exp['orientation'] = [0, 45, 90, 135]
exp['use keys']    = ['f', 'j']

# response mapping
# ----------------
resp = dict()
choose_resp = randint(0, 1)
resp[0]   = exp['use keys'][choose_resp]
resp[90]  = exp['use keys'][choose_resp]
resp[45]  = exp['use keys'][1 - choose_resp]
resp[135] = exp['use keys'][1 - choose_resp]
exp['keymap'] = resp
exp['choose_resp'] = choose_resp

# port settings
# -------------
portdict = dict()
portdict['send'] = exp['use trigger']
portdict['port address'] = int(exp['port address'], base=16) \
						   if exp['port address'] and portdict['send'] \
						   else exp['port address']
portdict['codes'] = {'fix': 1, 'mask': 2}
portdict['codes'].update({'target_' + str(ori) : 4 + i \
						   for i, ori in enumerate(exp['orientation'])
						   })
portdict['codes'].update({'breakStart': 100, 'breakStop': 102})
portdict['codes'].update({'training': 10, 'fitting': 20,
		'contrast': 30, 'time': 40})
exp['port'] = portdict

# subject info
# ------------
sub_data = getSubject()
if sub_data is None:
    core.quit()
exp['participant']  = dict()
exp['participant']['ID'] = sub_data[0]
exp['participant']['sex'] = sub_data[1][0]
# exp['participant']['age'] = sub_data[1]

# get path
pth   = os.path.dirname(os.path.abspath(__file__))
exp['path'] = pth

# ensure 'data' directory is available:
exp['data'] = os.path.join(pth, 'data')
if not os.path.isdir(exp['data']):
	os.mkdir(exp['data'])

# setup logging:
exp['logfile'] = os.path.join(exp['data'], exp['participant']['ID'] + '_c.log')

# check frame rate:
win = visual.Window(monitor="testMonitor")
exp['frm'] = getFrameRate(win)
win.close()

# create base dataframe
db = create_database(exp, trials=600)
