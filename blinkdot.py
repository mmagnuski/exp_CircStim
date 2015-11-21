# -*- coding: utf-8 -*-
# count flicker task - simple visual responses

# we need about 70 flicks (per condition if more than one)
# 5 - 15 flicks gives us average of 10 - so 7 trial should
# be ok
# 0.5 - 4 seconds before each flick, 1 second after

from psychopy import visual, event, core
from random import randint
import pandas as pd

# try importing ctypes.windll
try:
	from ctypes import windll
except:
	pass

# example sequence
frame_range = (70, 200)
trials_in_block = (5, 15)

def onflip_trigger(trig):
	windll.inpout32.Out32(trig[0], trig[1])

def dot(win, color=(1,1,1)):
    circ = visual.Circle(win, radius=0.15, edges=64, units='deg')
    circ.setFillColor(color)
    circ.setLineColor(color)
    return circ


def give_dot_stim(win):
	stim = dict()
	stim['dot_gray'] = dot(win, color = [0.3, 0.3, 0.3])
	stim['dot_white'] = dot(win, color = [1., 1., 1.])
	stim['text'] = visual.TextStim(win, pos=(0,6))
	stim['text type'] = visual.TextStim(win)
	return stim


def present_dot_trial(win, stim, times, trigger=False):
	for t in times:
		# wait for blink
		for _ in range(t[0]):
			stim['dot_gray'].draw()
			win.flip()
		# if trigger - onflip
		if trigger:
			win.callOnFlip(onflip_trigger, trigger)
		# present blink
		for _ in range(t[1]):
			stim['dot_white'].draw()
			win.flip()
		# clear port
		if trigger:
			win.callOnFlip(onflip_trigger, [trigger[0], 0])
		# post wait
		for _ in range(t[2]):
			stim['dot_gray'].draw()
			win.flip()

		if 'q' in event.getKeys():
			core.quit()


def get_num_resp(win, stim, trigger=False):
	finished_typing = False
	stim['text'].setText(u'Proszę wpisać ile razy mignęła kropka:')
	keys = list('0123456789') + ['return', 'backspace']
	typed_str = ''

	if trigger:
		win.callOnFlip(onflip_trigger, [trigger[0], 222])

	stim['text'].draw()
	win.flip()

	if trigger:
		win.callOnFlip(onflip_trigger, [trigger[0], 0])

	while not finished_typing:
		k = event.waitKeys(keyList=keys)
		if 'return' in k and typed_str:
			finished_typing = True
		elif 'backspace' in k and typed_str:
			typed_str = typed_str[:-1]
		elif 'q' in k:
			core.quit()
		else:
			typed_str += k[0]
		stim['text type'].setText(typed_str)
		stim['text type'].draw()
		stim['text'].draw()
		win.flip()
	return int(typed_str)


def give_trial_times(blink_lims, times):
	# testing:
	num_blinks = randint(*blink_lims)
	pre_rng = (times['pre min'], times['pre max'])
	return [ [randint(*pre_rng), times['stim'], 
		times['post']] for _ in range(num_blinks)]


def all_trials(win, stim, time, blink_lims=trials_in_block,
	min_blinks=100, trigger=False):
	num_all_blinks = 0
	tri = 0
	df = pd.DataFrame({'num_blinks' : [], 'num_typed' : [],
		'ifcorrect' : []})
	df = df.loc[:, ['num_blinks', 'num_typed', 'ifcorrect']]
	while num_all_blinks < min_blinks:
		# get wait times for each blink:
		trial_times = give_trial_times(blink_lims, time)

		# check how many blinks
		num_blinks = len(trial_times)
		print num_blinks
		num_all_blinks += num_blinks

		# present
		present_dot_trial(win, stim, trial_times, trigger=trigger)
		num_typed = get_num_resp(win, stim, trigger=trigger)
		df.loc[tri, 'num_blinks'] = num_blinks
		df.loc[tri, 'num_typed'] = num_typed
		df.loc[tri, 'ifcorrect'] = num_blinks == num_typed
		tri += 1
	return df


