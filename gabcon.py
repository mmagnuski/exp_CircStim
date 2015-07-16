# -*- coding: utf-8 -*-

# add description

# TODOs:
# [.] add instructions     (!)
# [ ] we need to log age and sex of the participant
#     (that would go to settings.py)
# [ ] add markers to:
#     -> start (and end?) of each break
#     ->
# [ ] test continue_dataframe for overwrite


# imports
# -------
from psychopy  import visual, core, event, logging

import os
import numpy  as np
import pandas as pd
from exputils  import (plot_Feedback, to_percent,
	round2step)
from weibull   import (fit_weibull,
	set_opacity_if_fit_fails, correct_Weibull_fit)
from stimutils import (exp, db, stim, startTrial,
	present_trial, present_break, show_resp_rules,
	present_feedback, present_training, textscreen,
	give_training_db, Instructions, Stepwise, trim)

if os.name == 'nt' and exp['use trigger']:
	from ctypes import windll

# set logging
lg = logging.LogFile(f=exp['logfile'], level=logging.WARNING, filemode='w')


# EXPERIMENT
# ==========

# INSTRUCTIONS!
instr = Instructions('instructions.yaml')
instr.present()

# show response rules:
show_resp_rules()

# TRAINING
# --------

if exp['run training']:
	# set things up
	slow = exp.copy()
	slow['opacity'] = [1.0, 1.0]
	txt = u'Twoja poprawność: {}\nOsiągnięto wymaganą poprawność.\n'
	addtxt = u'Szybkość prezentacji bodźców zostaje zwiększona.'
	for s, c in zip(exp['train slow'], exp['train corr']):
		current_corr = present_training(exp=slow, slowdown=s, corr=c)
		if s == 1:
			addtxt = 'Koniec treningu.'
		now_txt = txt + addtxt
		textscreen(now_txt.format(to_percent(current_corr)))
		show_resp_rules()


# ADD some more instructions here
# TODO - info that main experiment is about to begin

# Contrast fitting - stepwise
# ---------------------------

# init stepwise contrast adjustment
fitting_db = give_training_db(db, slowdown=1)
step = exp['step until']
s = Stepwise(corr_ratio=[1,1])
exp['opacity'] = [1., 1.]

while s.trial <= step[0] and len(s.reversals) < 3:
	present_trial(s.trial, db=fitting_db, exp=exp)
	stim['window'].flip()

	s.add(db.loc[i, 'ifcorrect'])
	c = s.next()
	exp['opacity'] = [c, c]

# more detailed stepping now
last_trial = s.trial
all_reversals = s.reversals[-1]
s = Stepwise(corr_ratio=[2,1], start=s.param, step=0.05)

while s.trial <= step[1]:
	present_trial(s.trial + last_trial, db=fitting_db, exp=exp)
	stim['window'].flip()

	s.add(db.loc[i, 'ifcorrect'])
	c = s.next()
	exp['opacity'] = [c, c]

all_reversals.append(s.reversals)
mean_thresh = np.mean(all_reversals)
tri = s.trial + last_trial


# Contrast fitting - weibull
# --------------------------

take_corr = [0.55, 0.65, 0.775, 0.9]
check_contrast = np.arange(mean_thresh-0.05, mean_thresh+0.1, 0.05)
while tri <= stim['fit until']:
	np.random.shuffle(check_contrast)
	for c in check_contrast:
		exp['opacity'] = [c, c]
		present_trial(tri, db=fitting_db, exp=exp)
		stim['window'].flip()
		tri += 1

	# fit weibull
	w = fitw(df, ind)
	# take threshold for specified correctness levels
	new_contrast = w.get_threshold(take_corr)
	if w.params[0] < 0.01:
		# if fit is not ok, move measurement points 0.05 down
		new_contrast = [c - 0.05 for c in new_contrast]
	# trim all points
	check_contrast = np.array([trim(c, exp['min opac'], 1.)
		for c in new_contrast])
	check_contrast = round2step(check_contrast)


# signal that main proc is about to begin
# ---------------------------------------
if exp['use trigger']:
	windll.inpout32.Out32(exp['port']['port address'], 255)
	core.wait(0.01)
	clear_port(exp['port'])


# MAIN EXPERIMENT
# ---------------

# main loop
for i in range(startTrial, exp['numTrials'] + 1):
	present_trial(i, exp=exp)
	stim['window'].flip()

	if step > 0 and i == step + 1:
		# add about 0.2 around the current contrast
		exp['opacity'] = [trim(contrast - 0.2,
							   exp['min opac'], 0.95),
						  trim(contrast + 0.2,
							   exp['min opac']+0.05, 1.)]

	# present break
	if (i) % exp['break after'] == 0:
		# save data before every break
		db.to_excel(os.path.join(exp['data'], exp['participant'] + '_c.xls'))
		# break and refresh keyboard mapping
		present_break(i)
		show_resp_rules()

	# TODO: close this into a def
	# fit Weibull function
	if i >= exp['fit from'] and i <= exp['fit until'] and \
		(i % exp['fit every']) == 0:

		# fitting psychometric function
		w = fit_weibull(db, i)
		newopac = w.get_threshold(exp['corrLims'])
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

db.to_excel(os.path.join(exp['data'], exp['participant'] + '_c.xls'))
core.quit()
