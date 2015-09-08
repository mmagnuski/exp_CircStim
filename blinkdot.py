# -*- coding: utf-8 -*-
# count flicker task - simple visual responses

# we need about 70 flicks (per condition if more than one)
# 5 - 15 flicks gives us average of 10 - so 7 trial should
# be ok
# 0.5 - 4 seconds before each flick, 1 second after

from psychopy import visual, event, core
from random import randrange

# example sequence
frame_range = (70, 200)
trials_in_block = (5, 15)


def give_dot_stim(win):
	stim = dict()
	stim['dot_gray'] = visual.Circle(win, color = [0.3, 0.3, 0.3])
	stim['dot_white'] = visual.Circle(win, color = [1., 1., 1.])
	stim['text'] = visual.TextStim(win, pos=(6,0))
	stim['text type'] = visual.TextStim(win)
	return stim


def present_dot_trial(win, stim, times):
	for t in times:
		for _ in t[0]:
			stim['dot_gray'].draw()
			win.flip()
		for _ in t[1]:
			stim['dot_white'].draw()
			win.flip()


def get_num_resp(win, stim):
	finished_typing = False
	stim['text'].setText(u'Proszę wpisać ile razy mignęła kropka:')
	keys = list('0123456789') + ['return', 'backspace']
	typed_str = ''
	while not finished_typing:
		k = event.waitKeys(keyList=keys)
		if 'return' in k and typed_str:
			finished_typing = True
		elif 'backspace' in k and typed_str:
			typed_str = typed_str[:-1]
		else:
			typed_str += k[0]
		stim['text type'].setText(typed_str)
		stim['text type'].draw()
		win.flip()
	return int(typed_str)


