from psychopy  import visual, core, event, logging

import os
import numpy  as np
import pandas as pd
import blinkdot as blnk
from exputils  import (DataManager, ms2frames, getFrameRate)
from stimutils import (exp, stim, Instructions, 
	onflip_work, clear_port)

if os.name == 'nt' and exp['use trigger']:
	from ctypes import windll

# set logging
dm = DataManager(exp)
exp = dm.update_exp(exp)
exp['numTrials'] = 500 # ugly hack, change
log_path = dm.give_path('l', file_ending='log')
lg = logging.LogFile(f=log_path, level=logging.WARNING, filemode='w')


# blinking dot
# ------------
if exp['run blinkdot']:
	stim['window'].blendMode = 'avg'
	instr = Instructions('dot_instructions.yaml')
	instr.present()
	stim['window'].blendMode = 'avg'
	frms = getFrameRate(stim['window'])
	dotstim = blnk.give_dot_stim(stim['window'])

	# times in ms to get trial times
	time = dict()
	time['pre min'] = 500
	time['pre max'] = 4000
	time['stim'] = 100
	time['post'] = 1000

	trigger = False
	if exp['use trigger']:
		trigger = [exp['port address'], 123]

	time = ms2frames(time, frms['time'])
	df = blnk.all_trials(stim['window'], dotstim, time,
		trigger=trigger)
	df.to_excel(dm.give_path('0'))
	stim['window'].blendMode = 'add'