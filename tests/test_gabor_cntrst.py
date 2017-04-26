'''Tests different gabor contrasts for visibility along with blend mode add
and overlaying gabors on top of each other if contrast > 1. Offers simple
interface to control contrast.'''

from psychopy import core, visual, event

win = visual.Window(monitor='testMonitor')
# win = visual.Window(monitor='testMonitor', blendMode='add', useFBO=True)

def gabor(win=win, ori=0, opa=1.0, pos=[0, 0], size=5, c=1.0, sf=2,
		  units='deg'):
	return visual.GratingStim(win=win,  mask="gauss", size=size, pos=pos,
 							  sf=sf, ori=ori, opacity=opa, units=units,
							  contrast=c)

# start values
opacity = 1.
contrast = 1.
step = 0.005
ifCnt = True
txt = visual.TextStim(win, text = str(opacity),
	pos = (0., 0.85))
txt2 = visual.TextStim(win, text = str(opacity),
	pos = (0., 0.7))
txt3 = visual.TextStim(win, text = str(contrast),
	pos = (0., 0.55))
g = gabor(opa = opacity)
# g2 = gabor(opa = opacity)

while ifCnt:
	# if contrast > 1.:
	# 	g2.draw()
	g.draw()
	txt.draw()
	# txt2.draw()
	txt3.draw()
	win.flip()

	k = event.waitKeys()
	print k
	if 'q' in k:
		ifCnt = False
		break
	elif 'up' in k:
		opacity += step
	elif 'down' in k:
		opacity -= step
	elif 'minus' in k:
		step -= 0.001
	elif 'equal' in k:
		step += 0.001
	# elif 'c' in k:
	# 	contrast += step
	# elif 'v' in k:
	# 	contrast -= step

	g.opacity = opacity
	# g2.opacity = opacity
	# if contrast <= 1.:
	# 	g.contrast = contrast
	# else:
	# 	g.contrast = 1.0
	# 	g2.contrast = contrast - 1.
	txt.setText('opacity: ' + str(opacity))
	# txt2.setText('contrast: ' + str(contrast))
	txt3.setText('step size: ' + str(step))

core.quit()
