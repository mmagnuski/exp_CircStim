from psychopy import visual, event

window_1 = visual.Window(screen=0, fullscr=True, monitor='testMonitor', blendMode='avg')
window_2 = visual.Window(screen=1, fullscr=True, monitor='testMonitor', blendMode='add',
                         useFBO=True)

text1 = visual.TextStim(window_1, text='1')
text2 = visual.TextStim(window_2, text='2')

text1.draw()
text2.draw()
window_1.flip()
window_2.flip()
event.waitKeys()