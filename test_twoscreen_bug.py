# test two screens
from psychopy import visual, event

window_1 = visual.Window(screen=0, fullscr=True, monitor='testMonitor', blendMode='avg')
window_2 = visual.Window(screen=1, fullscr=True, monitor='testMonitor', blendMode='avg')

text1 = visual.TextStim(window_1, text='1')
text2 = visual.TextStim(window_2, text='2')

for idx in range(2):
	text1.draw()
	text2.draw()
	window_1.flip()
	window_2.flip()
	event.waitKeys()

# window_2.getMovieFrame()
# window_2.saveMovieFrames('some_image.png')
# print(window_1.blendMode)
# print(window_2.blendMode)
# print(text1.win)
# print(text2.win)

# img = visual.ImageStim(window_1, 'some_image.png', pos=(0., -0.5), size=(0.6, 0.6))

for idx in range(2):
	text1.draw()
	# img.draw()
	text2.draw()
	window_1.flip()
	window_2.flip()
	event.waitKeys()