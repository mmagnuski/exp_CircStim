import os
import pandas as pd
from psychopy import visual, event, core
from stimutils import stim, exp
from exputils import ContrastInterface

# edit stim
stim['centerImage'] = visual.ImageStim(stim['window'],
    image=r'C:\proj\src\granprel\data\beh\pretest 01\T004_weibull_fit_temp.png', units='pix')

# read data
pth = r'C:\proj\src\granprel\data\beh\pretest 02'
fl = 'pretest_02_001b_b_1.xls'
flpth = os.path.join(pth, fl)
df = pd.read_excel(flpth)

if not 'window2' in stim:
    stim['window'].blendMode = 'avg'

interf = ContrastInterface(stim=stim, exp=exp, df=df)
interf.loop()
