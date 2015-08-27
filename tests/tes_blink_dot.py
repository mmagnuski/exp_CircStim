# count flicker task - simple visual responses

from psychopy import visual, event, core

win = visual.Window(monitor='testMonitor', units='deg')
dot_gray = visual.Circle(win, fillColor = [0.3, 0.3, 0.3], radius=0.2)
dot_white = visual.Circle(win, fillColor = [1., 1., 1.], radius=0.2)

# example sequence
times = [(100, 4), (58, 4), (72, 4), (120, 4), (88, 4)]
for t in times:
	for _ in range(t[0]):
		dot_gray.draw()
		win.flip()
	for _ in range(t[1]):
		dot_white.draw()
		win.flip()
        
text = visual.TextStim(win, text=u'Press key')
text.draw(); win.flip()
k = ''
str_num = ''
while 'q' not in k:
    k = event.waitKeys(list('1234567890q') + ['backspace'])
    if 'backspace' in k and len(str_num) > 0:
        str_num = str_num[:-1]
    else:
        str_num += k[0]
    text.setText(str_num)
    text.draw(); win.flip()
core.quit()