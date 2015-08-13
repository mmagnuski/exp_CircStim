from exputils import create_database
import numpy as np

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
print '100 trials:'
print df.head(n=10)
print df.shape

contrast = np.linspace(0.1, 0.5, num=10)
df = create_database(exp, rep=2, combine_with=('opacity', contrast))
print '2 repetitions of combinations with contrast'
print 'contast: ', contrast
print df.head(n=20)
print df.shape