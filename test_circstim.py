from   circutils import CircStim
from   psychopy  import core, visual, event
from   random    import choice, uniform
import pandas    as     pd

frms = [6, 4, 20]

# a list of all possible could be created before trials
params = {}
params['circleRadius']     = 0.4
params['circleResolution'] = 5
params['p1']               = ['+', 'x']
params['p2']               = ['d', 'l']

# DEFs
# ----
def draw_stim(stims, frms, win):
    for f in range(frms):
        for s in stims:
            s.draw()
        win.flip()

def get_stims(params):
    trg = CircStim(r = params['circleRadius'], n = params['circleResolution'], 
                    opacity = uniform(0.1, 1), 
                    pattern = choice(params['p1']) + choice(params['p2']), 
                    window = win)
    msk = CircStim(r = params['circleRadius'], n = params['circleResolution'], 
                    pattern = 'm' + choice(params['p2']), 
                    window = win)
    return trg, msk
        

win = visual.Window()


for m in mask:
    m.create_shapes() 
for t in trg:
    t.create_shapes() # this will later be automatically called in CircStim.__init__

# wait a while
core.wait(1)

# present stims
stims = [choice(trg), mask[0], mask[1]]


