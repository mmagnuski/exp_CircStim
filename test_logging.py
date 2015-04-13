# test interactions between logging module and psychopy
# psychopy logging is a little weird - seems to be less flexible
# than standard library logging

import os
from psychopy import logging


pth = r'D:\DATA\EXPERIMENTS\GabCon 2015\data'
logfile = os.path.join(pth, 'test.log')

# this does not work with psychopy logging
# logging.basicConfig(filename=logfile, level=logging.INFO)

# this does not too:
# (import logging; from psychopy import logging as log)
# logger = logging.getLogger(__name__)
# log.LogFile(f=logfile, level=0, filemode='w', logger=logger)

# log now
lg = logging.LogFile(f=logfile, level=0, filemode='w')
logging.log(logging.INFO, 'This is the first message')
logging.log(logging.INFO, 'Another message')
logging.log(logging.INFO, 'Hope it works...')

logging.flush()