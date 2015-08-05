from psychopy import core, visual, event

win = visual.Window(monitor='testMonitor', blendMode='add', useFBO=True)

g1 = visual.GratingStim(win=win, mask='gauss', sf=5, pos=[-0.6, 0.], contrast=1.0)
g2 = visual.GratingStim(win=win, mask='gauss', sf=5, pos=[0.6, 0.], contrast=.7)
g3 = visual.GratingStim(win=win, mask='gauss', sf=5, pos=[0.6, 0.], contrast=.7)
g4 = visual.GratingStim(win=win, mask='gauss', sf=5, pos=[0.6, 0.], contrast=.7)
[x.draw() for x in [g1, g2, g3, g4]]
win.flip()

event.waitKeys()
core.quit()
