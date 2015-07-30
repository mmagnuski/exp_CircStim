from psychopy import core, visual, event

# create two windows - one for each monitor
win1 = visual.Window(monitor='testMonitor', )

g1 = visual.GratingStim(win=win, mask='gauss', sf=5, pos=[-0.6, 0.], contrast=1.0)
g2 = visual.GratingStim(win=win, mask='gauss', sf=5, pos=[0.6, 0.], contrast=.5)