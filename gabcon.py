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
from weibull   import (fit_weibull, set_opacity_if_fit_fails,
					  correct_Weibull_fit)
from stimutils import (exp, db, stim, startTrial, present_trial,
					  present_break, show_resp_rules,
					  present_feedback, give_training_db,
					  Instructions, Stepwise, trim)


# set logging
lg = logging.LogFile(f=exp['logfile'], level=logging.INFO, filemode='w')


# EXPERIMENT
# ==========

# INSTRUCTIONS!
instr = Instructions('instructions.yaml')
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
if exp['use trigger']:
	windll.inpout32.Out32(exp['port']['port address'], 255)
	core.wait(0.01)
	clear_port(exp['port'])

# ADD some more instructions here
# TODO - info that main experiment is about to begin
show_resp_rules()


# MAIN EXPERIMENT
# ---------------

step = exp['step until']

# init stepwise contrast adjustment
if exp['step until'] > 0:
	s = Stepwise()
	exp['opacity'] = [1., 1.]

# main loop
for i in range(startTrial, exp['numTrials'] + 1):
	present_trial(i, exp=exp)
	stim['window'].flip()

	if (i) <= step:
		s.add(db.loc[i, 'ifcorrect'])
		contrast = s.next()
		exp['opacity'] = [contrast, contrast]

	if step > 0 and i == step + 1:
		# add about 0.2 around the current contrast
		exp['opacity'] = [trim(contrast - 0.2,
							   exp['min opac'], 0.95),
						  trim(contrast + 0.2,
							   exp['min opac']+0.05, 1.)]

	# present break
	if (i) % exp['break after'] == 0:
		# save data before every break
		db.to_excel(os.path.join(exp['data'], exp['participant'] + '.xls'))
		# break and refresh keyboard mapping
		present_break(i)
		show_resp_rules()

	# TODO: close this into a def
	# fit Weibull function
	if i >= exp['fit from'] and i <= exp['fit until']:

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

	# inter-trial interval
	stim['window'].flip()
	core.wait(0.5) # pre-fixation time is always the same

db.to_excel(os.path.join(exp['data'], exp['participant'] + '.xls'))
core.quit()
