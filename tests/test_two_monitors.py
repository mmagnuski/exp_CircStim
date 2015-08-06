# test for two monitors:
import time
from psychopy.visual import Window
from psychopy.visual import TextStim
from psychopy import event

# create a Window
create_win = lambda s: Window(size=(1280,1024), fullscr=True,
	monitor="testMonitor", winType='pyglet', screen=s)
win1 = create_win(0)
win2 = create_win(1)
# draw some text on the Window
txt = list()
txt.append(TextStim(win1, text="This is screen 1"))
txt.append(TextStim(win2, text="This is screen 2"))
txt2 = TextStim(win2, pos=(0, -0.5), text="another text")
[t.draw() for t in txt]
win1.flip()
event.waitKeys()
win2.flip(clearBuffer=False)
event.waitKeys()
txt2.draw()
win2.flip(clearBuffer=False)
event.waitKeys()
