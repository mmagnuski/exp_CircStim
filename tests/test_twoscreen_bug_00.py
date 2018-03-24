from psychopy import visual, event
import numpy as np
import matplotlib.pyplot as plt

window = visual.Window(monitor='testMonitor')
text = visual.TextStim(window)

def show_all(idx): 
	text.setText(str(idx))
	text.draw()
	window.flip()
	event.waitKeys()

show_all(0) # 0 is visible, press any key

# matplotlib draws a simple plot
plt.plot(np.arange(10), np.random.rand(10).cumsum())
# window.winHandle.activate()

show_all(1) # you see 1 but keyboard does not work
# click anywhere in the window and the keyboard is working again