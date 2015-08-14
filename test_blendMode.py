from psychopy import visual, event, core

win = visual.Window(monitor='testMonitor', units='deg', 
    useFBO=True, blendMode='add')

gab = visual.GratingStim(win, mask='gauss', ori=0,
    sf=1.5, size=5, pos=(-8, 0))
tex = visual.TextStim(win, text="Text, text, text", pos=(0, 5))
present_oreder = [[gab], [gab, tex], [gab, tex]]
for stims in present_oreder:
	for s in stims:
		s.draw()
	win.flip()
	event.waitKeys()
