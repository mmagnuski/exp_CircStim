import os
import pandas as pd
from psychopy import visual, event, core
from stimutils import stim, exp
from exputils import ContrastInterface, get_valid_path

# edit stim
root_dir = get_valid_path([r'D:\DATA\granprel\data\beh',
						   r'C:\proj\src\granprel\data\beh'])
img_pth = os.path.join(root_dir, r'pretest 01\T004_weibull_fit_temp.png')
stim['centerImage'] = visual.ImageStim(stim['window'], image=img_pth,
									   units='pix')

# read data
df_pth = os.path.join(root_dir, 'pretest 02')
fl = 'pretest_02_001b_b_1.xls'
flpth = os.path.join(pth, fl)
df = pd.read_excel(flpth)

if not 'window2' in stim:
    stim['window'].blendMode = 'avg'

interf = ContrastInterface(stim=stim, exp=exp, df=df)
interf.loop()
