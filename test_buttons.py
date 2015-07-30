from psychopy import visual, event, core
from exputils import Button
import numpy as np

win = visual.Window(monitor="testMonitor")
button_pos = np.zeros([3,2])
button_pos[:,0] = 0.5
button_pos[:,1] = [0.5, 0., -0.5]
button_text = list('ABC')
buttons = [Button(win=win, pos=p, text=t, size=(0.5,0.15))
	for p, t in zip(button_pos, button_text)]
mouse = event.Mouse()

[b.draw() for b in buttons]
win.flip()

# wait till mouse click is on text:
clicks = 0
while clicks < 4:
	m1, m2, m3 = mouse.getPressed()
	if m1:
		mouse.clickReset()
		ifclicked = [b.contains(mouse) for b in buttons]
		which_clicked = np.where(ifclicked)[0]
		if which_clicked.size > 0:
			buttons[which_clicked[0]].click()
			[b.draw() for b in buttons]
			win.flip()
			clicks += 1
		print clicks
		core.wait(0.1)
core.quit()
