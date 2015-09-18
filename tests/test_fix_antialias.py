from psychopy import visual, event

win = visual.Window(monitor='testMonitor')

def fix(win=win, color=(0.5, 0.5, 0.5), edges=16, interp=True):
	dot = visual.Circle(win, radius=0.15,
		edges=edges, units='deg', interpolate=interp)
	dot.setFillColor(color)
	dot.setLineColor(color)
	return dot

blend = ["avg", "avg", "add", "add"]
intrp = [False, True, False, True]

for b, i in zip(blend, intrp):
	win.blendMode = b
	f = fix(interp=i)
	f.draw()
	win.flip()
	event.waitKeys()