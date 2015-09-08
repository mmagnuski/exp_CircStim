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
	stim['text'] = visual.TextStim(win, pos=(6,0))
	stim['dot_gray'] = dot(win, color = [0.3, 0.3, 0.3])
	stim['dot_white'] = dot(win, color = [1., 1., 1.])
	stim['text type'] = visual.TextStim(win)
	return stim


def present_dot_trial(win, stim, times, trigger=False):
	for t in times:
		for _ in t[0]:
			stim['dot_gray'].draw()
			win.flip()
		for _ in t[1]:
		# if trigger - onflip
		if trigger:
			win.callOnFlip(onflip_trigger, trigger)
			stim['dot_white'].draw()
			win.flip()
		# clear port
		if trigger:
			win.callOnFlip(onflip_trigger, [trigger[0], 0])


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
		else:
			typed_str += k[0]
		stim['text type'].setText(typed_str)
		stim['text type'].draw()
		win.flip()
	return int(typed_str)


