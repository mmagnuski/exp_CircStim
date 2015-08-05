import os

# set dir
os.chdir(r'D:\DATA\experiments\gabcon 2015')

from exputils import create_database

# construct exp
exp = dict()
exp['targetTime']  = [1]
exp['SMI']         = [2]
exp['fixTimeLim']  = [0.75, 2.5]
exp['maskTime']    = [20]
exp['fdb time']    = [40]
exp['orientation'] = [0, 45, 90, 135]
exp['frm'] = dict()
exp['frm']['time'] = 10

df = create_database(exp, trials=100)