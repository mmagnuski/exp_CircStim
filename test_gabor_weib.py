# -*- coding: utf-8 -*-

# add description

# TODOs:
# [ ] add instructions     (!)
# [ ] add training         (!)
#     -> +slowdown +full-contrast?
#     -> +feedback
# [ ] add markers to:
#     -> start (and end?) of each break
#     -> 
# [ ] remove print statements and add logging to a file - 
#     useful in inspecting how fit and overcoming fit 
#     failures works
# [ ] test continue_dataframe for overwrite
# [ ] modularize and organize code
# [x] use '\data' folder to save data
# [x] test RT measurement on some platforms (timestamping
#         may not work...)
# [x] reset timer on stim presentation with callOnFlip
# [x] send LPT triggers with callOnFlip
# [x] add weibull fitting on fixed frame setup (1-1 frms)
# 
# not necessary:
# [ ] load seaborn conditionally
# [ ] ? present some scatter feedback
# [ ] ? check interpolate = True in visual.ImageStim

# imports
print 'importing psychopy...'
from psychopy  import visual, core, event
from exputils  import Weibull, plot_Feedback
from random    import randint, uniform #, choice
from stimutlis import exp, db, startTrial, present_trial, \
					  present_break, show_resp_rules
import os
import numpy  as np
import pandas as pd


# some remaining defs:
def set_opacity_if_fit_fails(corr, exp):
	mean_corr = np.mean(corr)
	if mean_corr > 0.8:
		exp['opacity'][0] *= 0.5
		exp['opacity'][1] *= 0.5 
		exp['opacity'][0] = np.max([exp['opacity'][0], 0.01])
	elif mean_corr < 0.6:
		exp['opacity'][0] = np.min([exp['opacity'][1]*2, 0.8])
		exp['opacity'][1] = np.min([exp['opacity'][1]*2, 1.0])


def fit_weibull(db, i):
#	if i < 40:
#		idx = np.array(np.linspace(1, i, num = i), dtype = 'int')
#	else:
#		idx = np.array(np.linspace(i-40+1, i, num = 40), dtype = 'int')
	idx = np.array(np.linspace(1, i, num = i), dtype = 'int')
	ifcorr = db.loc[idx, 'ifcorrect'].values.astype('int')
	opacit = db.loc[idx, 'opacity'].values.astype('float')

	# fit on non-nan
	notnan = ~(np.isnan(ifcorr))
	w = Weibull(opacit[notnan], ifcorr[notnan])
	w.fit([1.0, 1.0])
	return w


# start experiment
# ----------------

# INSTRUCTIONS!

# show response rules:
show_resp_rules()

# trial loop:
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
		if i < 101:
			w = fit_weibull(db, i)
			print 'Weibull params: ', w.params
			
			newopac = w._dist2corr(exp['corrLims'])
			# TODO this needs checking, removing duplicates and testing
			if newopac[1] < 0.005 or newopac[1] <= newopac[0] or w.params[0] < 0 \
				or newopac[1] < 0.01 or newopac[0] > 1.0:
				set_opacity_if_fit_fails(w.orig_y, exp)
			else:
				exp['opacity'] = newopac
            
			if exp['opacity'][1] > 1.0:
				exp['opacity'][1] = 1.0
			if exp['opacity'][0] < 0.01:
				exp['opacity'][0] = 0.01
			if exp['opacity'][0] > exp['opacity'][1]:
				exp['opacity'][0] = exp['opacity'][1]/2

			# DEBUG
			print 'opacity limits set to: ', exp['opacity']

			# show weibull fit
			plot_Feedback(stim, w, exp['data'])

		# break and refresh keyboard mapping
		present_break(i)
		show_resp_rules()

	stim['window'].flip()
	core.wait(0.5) # pre-fixation time is always the same

db.to_excel(os.path.join(exp['data'], exp['participant'] + '.xls'))
core.quit()