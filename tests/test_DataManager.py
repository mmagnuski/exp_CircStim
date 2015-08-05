from settings import exp
from exputils import DataManager

print 'keymap before data manager: ', exp['keymap']
dm = DataManager(exp)
print 'keymap after data manager: ', dm.keymap

pth = dm.give_path('a')
print 'path of type "a": ', pth