# -*- coding: utf-8 -*-

# add description

# TODOs:
# [.] add instructions     (!)
# [x] we need to log age and sex of the participant
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
from weibull   import fitw
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

# INSTRUCTIONS
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

while s.trial <= step[0] and len(s.reversals) < 5:
	present_trial(s.trial, db=fitting_db, exp=exp)
	stim['window'].flip()

	s.add(fitting_db.loc[s.trial, 'ifcorrect'])
	c = s.next()
	exp['opacity'] = [c, c]

# more detailed stepping now
last_trial = s.trial - 1
start_param = np.mean(s.reversals) if \
	len(s.reversals) > 1 else s.param
s = Stepwise(corr_ratio=[2,1], start=s.param, vmin=0.05,
	step=0.05)

while s.trial <= step[1]:
	trial = s.trial + last_trial
	present_trial(trial, db=fitting_db, exp=exp)
	stim['window'].flip()

	s.add(fitting_db.loc[trial, 'ifcorrect'])
	c = s.next()
	exp['opacity'] = [c, c]

mean_thresh = np.mean(s.reversals) if s.reversals else c
# save fitting dataframe
fitting_db.to_excel(os.path.join(exp['data'], exp['participant']['ID'] + '_b.xls'))


# Contrast fitting - weibull
# --------------------------
trial += 1
params = [1., 1.]
check_contrast = np.arange(mean_thresh-0.05, mean_thresh+0.1, 0.05)
while trial <= exp['fit until']:
	np.random.shuffle(check_contrast)
	for c in check_contrast:
		exp['opacity'] = [c, c]
		present_trial(trial, db=fitting_db, exp=exp)
		stim['window'].flip()
		trial += 1

	# fit weibull
	take_last = min(trial-10, 50)
	ind = np.r_[trial-take_last:trial] # because np.r_ does not include last value
	w = fitw(fitting_db, ind, init_params=params)
	params = w.params

	# take threshold for specified correctness levels
	if w.params[0] <= 0.01:
		contrast_range = [exp['min opac'], 0.2]
	else:
		contrast_range = w.get_threshold(exp['corrLims'])

	# get 5 values from the contrast range
	check_contrast = np.linspace(contrast_range[0], 
		contrast_range[1], num=5)
	# trim all points
	check_contrast = np.array([trim(c, exp['min opac'], 1.)
		for c in check_contrast])
	check_contrast = round2step(check_contrast)

	# show weibull fit
	if exp['debug']:
		plot_Feedback(stim, w, exp['data'])

# save fitting dataframe!
fitting_db.to_excel(os.path.join(exp['data'], exp['participant']['ID'] + '_b.xls'))


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
		db.to_excel(os.path.join(exp['data'], exp['participant']['ID'] + '_c.xls'))
		# break and refresh keyboard mapping
		present_break(i)
		show_resp_rules()

	# inter-trial interval
	stim['window'].flip()
	core.wait(0.5) # pre-fixation time is always the same

db.to_excel(os.path.join(exp['data'], exp['participant']['ID'] + '_c.xls'))
core.quit()
