import os
import pandas as pd
from psychopy import visual, event, core
from stimutils import stim, exp
from exputils import ContrastInterface

# edit stim
stim['centerImage'] = visual.ImageStim(stim['window'],
    image=r'data\weibull_fit_temp.png', units='pix')

# read data
pth = exp['data']
# fl = 'new_test_01_b_1.xls'
# fl = 'Alexis01_b.xls'
# fl = 'T19_b_1.xls'
fl = 'testing_miko_01_b_1.xls'
flpth = os.path.join(pth, fl)
df = pd.read_excel(flpth)

if not 'window2' in stim:
    stim['window'].blendMode = 'avg'

interf = ContrastInterface(stim=stim, exp=exp, df=df)
interf.loop()
