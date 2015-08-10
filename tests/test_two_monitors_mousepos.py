# test for two monitors:
import time
from psychopy.visual import Window
from psychopy.visual import TextStim
from psychopy import event

def waitmouse(mouse):
	while True:
		m1, m2, m3 = mouse.getPressed()
		if m1:
			return m1

# create a Window
create_win = lambda s: Window(size=(1280,1024), fullscr=True,
	monitor="testMonitor", screen=s)
win1 = create_win(0)
win2 = create_win(1)
mouse = event.Mouse(win=win2)
# draw some text on the Window
txt = list()
txt.append(TextStim(win1, text="This is screen 1"))
txt.append(TextStim(win2, text="This is screen 2"))

while True:
	for t in txt:
		t.draw()	
	win1.flip()
	win2.flip()
	waitmouse(mouse)
	pos = mouse.getPos()
	for t in txt:
		t.setText(str(pos))

	k = event.getKeys()
	if 'q' in k:
		break
