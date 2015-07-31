from psychopy import visual, event, core
from exputils import ContrastInterface

stim = dict()
stim['window'] = visual.Window(monitor='testMonitor', fullscr=True)
stim['centerImage'] = visual.ImageStim(stim['window'], image=r'data\T004_weibull_fit_temp.png')

interf = ContrastInterface(stim=stim)

while True:
	interf.refresh()
	k = event.getKeys()
	if k and 'q' in k:
		core.quit()
