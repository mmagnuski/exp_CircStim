from psychopy import visual, event, core

win = visual.Window(monitor='testMonitor', fullscr=True)
img = visual.ImageStim(win, image=r'data\T004_weibull_fit_temp.png')
print 'initial img units: ', img.units
print 'initial img size: ', img.size

img.draw()
win.flip()

img.units = 'pix'
print 'units changed to pix'
print 'img units: ', img.units
print 'img size: ', img.size

