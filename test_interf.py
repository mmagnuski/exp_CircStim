from psychopy import visual, event, core
from exputils import ContrastInterface

stim = dict()
stim['window'] = visual.Window(monitor='testMonitor', fullscr=True)
stim['centerImage'] = visual.ImageStim(stim['window'],
    image=r'data\weibull_fit_temp.png', units='pix')

interf = ContrastInterface(stim=stim)
interf.loop()
