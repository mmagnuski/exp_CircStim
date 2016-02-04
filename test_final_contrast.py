from __future__ import print_function

import os
import pandas as pd
from psychopy import visual, event, core

from stimutils import stim, exp
from exputils import FinalFitGUI
from weibull import fitw


# read data
pth = exp['data']
# fl = 'new_test_01_b_1.xls'
fl = 'Alexis01_b.xls'
flpth = os.path.join(pth, fl)
df = pd.read_excel(flpth)


if not 'window2' in stim:
    stim['window'].blendMode = 'avg'

stim['target'][0].draw()
stim['window'].flip()
core.wait(.15)


# stim['centerImage'] = visual.ImageStim(stim['window'], image=r'data\T004_weibull_fit_temp.png')
fgui = FinalFitGUI(exp=exp, stim=stim, db=df, fitfun=fitw)
fgui.refresh_weibull()
fgui.GUI_loop()