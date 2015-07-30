from psychopy import visual, event, core
from exputils import ClickScale
import numpy as np

win = visual.Window(monitor="testMonitor")
scale = ClickScale(win=win, size=(0.6, 0.15))
mouse = event.Mouse()

scale.draw()
win.flip()

# wait till mouse click is on text:
clicks = 0
while clicks < 4:
	m1, m2, m3 = mouse.getPressed()
	if m1:
		mouse.clickReset()
		scale.test_click(mouse)	
		scale.draw()
		win.flip()
		clicks += 1
		print scale.points
		core.wait(0.1)
	elif m3:
		mouse.clickReset()
		scale.remove_point(-1)
		scale.draw()
		win.flip()
		print scale.points
		core.wait(0.1)
core.quit()
