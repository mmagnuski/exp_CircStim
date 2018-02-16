
First time - 1 only blinks on the screen 0, then the whole screen is white.

```python
from psychopy import visual, event

window_1 = visual.Window(screen=0, fullscr=True, monitor='testMonitor')
window_2 = visual.Window(screen=1, fullscr=True, monitor='testMonitor')

text1 = visual.TextStim(window_1, text='1')
text2 = visual.TextStim(window_2, text='2')

for idx in range(2):
	text1.draw()
	text2.draw()
	window_1.flip()
	window_2.flip()
	event.waitKeys()
```