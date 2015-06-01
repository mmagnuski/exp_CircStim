# -*- coding: utf-8 -*-

# add description

# TODOs:
# [ ] add instructions     (!)
# [ ] change logging level to WARNING
# [ ] add stepwise constrast checking
# [ ] we need to log age and sex of the participant
#     (that would go to settings.py)
# [ ] add markers to:
#     -> start (and end?) of each break
#     -> 
# [ ] test continue_dataframe for overwrite
# 
# not necessary:
# [ ] load seaborn conditionally

# imports
# -------
from psychopy  import visual, core, event, logging

import os
import numpy  as np
import pandas as pd
from ctypes    import windll
from exputils  import plot_Feedback
from weibull   import fit_weibull, set_opacity_if_fit_fails \
					  correct_Weibull_fit
from stimutils import exp, db, stim, startTrial, present_trial, \
					  present_break, show_resp_rules, \
					  present_feedback, give_training_db, \
					  Instructions


# set loggingly logging to logfile
lg = logging.LogFile(f=exp['logfile'], level=logging.INFO, filemode='w')


# EXPERIMENT
# ==========

# INSTRUCTIONS!
instr = Instructions('instructions.yml')
instr.present()

# show response rules:
show_resp_rules()

# SLOW TRAINING
# -------------

# set things up
slow = exp.copy()
slow['opacity'] = [0.7, 1.0]
train_db = give_training_db(db, slowdown=5)

i = 1
training_correctness = 0
while training_correctness < exp['train corr'][0] or i < 14:
	present_trial(i, exp=slow, db=train_db)

	# feedback:
	present_feedback(i, db=train_db)
	# check correctness
	training_correctness = train_db.loc[1:i, 'ifcorrect'].mean()
	i += 1


# signal that main proc is about to begin
# ---------------------------------------
windll.inpout32.Out32(exp['port']['port address'], 255)
show_resp_rules()

# TODO - info that main experiment is about to begin

# MAIN EXPERIMENT
# ---------------
for i in range(startTrial, exp['numTrials'] + 1):
	present_trial(i)
	stim['window'].flip()

	# present break 
	if (i) % exp['break after'] == 0:
		# save data before every break
		db.to_excel(os.path.join(exp['data'], exp['participant'] + '.xls'))

		# TODO: close this into a def
		# if break was within first 100 trials,
		# fit Weibull function
		if i <= exp['fit until']:

			# fitting psychometric function
			w = fit_weibull(db, i)
			newopac = w._dist2corr(exp['corrLims'])
			exp, logs = correct_Weibull_fit(w, exp, newopac)

			# log messages
			for log in logs:
				logging.warning(log)
			logging.flush()

			# show weibull fit
			plot_Feedback(stim, w, exp['data'])

		# break and refresh keyboard mapping
		present_break(i)
		show_resp_rules()

	stim['window'].flip()
	core.wait(0.5) # pre-fixation time is always the same

db.to_excel(os.path.join(exp['data'], exp['participant'] + '.xls'))
core.quit()