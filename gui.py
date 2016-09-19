from psychopy import visual

class Button:
	'''
	Simple button class, does not check itself, needs to be checked
	in some event loop.

	create buttons
	--------------
	win = visual.Window(monitor="testMonitor")
	button_pos = np.zeros([3,2])
	button_pos[:, 0] = 0.5
	button_pos[:, 1] = [0.5, 0., -0.5]
	button_text = list('ABC')
	buttons = [Button(win=win, pos=p, text=t) for p, t in
			   zip(button_pos, button_text)]

	draw buttons
	------------
	[b.draw() for b in buttons]
	win.flip()

	check if buttons were pressed
	-----------------------------
	mouse = event.Mouse()
	m1, m2, m3 = mouse.getPressed()
	if m1:
		mouse.clickReset()
		ifclicked = [b.contains(mouse) for b in buttons]
		which_clicked = np.where(ifclicked)[0]
		if which_clicked.size > 0:
			buttons[which_clicked[0]].click()
	'''

	def __init__(self, pos=(0, 0), win=None, size=(0.4, 0.15),
		text='...', box_color=(-0.3, -0.3, -0.3), font_height=0.08,
		units='norm', click_color=(0.2, -0.3, -0.3)):
		self.rect_stim = visual.Rect(win, pos=pos, width=size[0],
			height=size[1], fillColor=box_color, lineColor=box_color,
			units=units)
		self.text_stim = visual.TextStim(win, text=text, pos=pos,
			height=font_height, units=units)
		self.clicked = False
		self.orig_color = box_color
		self.click_color = click_color
		self.click_fun = self.default_click_fun

	def draw(self):
		self.rect_stim.draw()
		self.text_stim.draw()

	def contains(self, obj):
		return self.rect_stim.contains(obj)

	def setText(self, text):
		self.text_stim.setText(text)

	def default_click_fun(self):
		if self.clicked:
			self.clicked = False
			self.rect_stim.setFillColor(self.orig_color)
			self.rect_stim.setLineColor(self.orig_color)
		else:
			self.clicked = True
			self.rect_stim.setFillColor(self.click_color)
			self.rect_stim.setLineColor(self.click_color)

	def click(self):
		self.click_fun()


class ClickScale(object):

	def __init__(self, win=None, size=(0.5, 0.15), pos=(0, 0),
		color=(-0.3, -0.3, -0.3), units='norm', lims=[0., 1.]):
		self.win = win
		# maybe - raise ValueError if win is none
		self.points = []
		self.lines = []
		self.units = units
		self.color = color
		self.pos = pos
		self.lims = lims
		self.length = lims[1] - lims[0]
		self.line_color = (1.0, -0.3, -0.3)
		self.scale = visual.Rect(win, pos=pos, width=size[0],
				height=size[1], fillColor=color, lineColor=color,
				units=units)
		# TODO: check if pos is divided by two
		self.x_extent = [pos[0]-size[0]/2., pos[0]+size[0]/2.]
		self.x_len = size[0]
		self.h = size[1]

	def point2xpos(self, point):
		return (self.x_extent[0] + (point - self.lims[0]) / self.length *
				self.x_len)

	def xpos2point(self, xpos):
		return (self.lims[0] + (xpos - self.x_extent[0]) / self.x_len *
				self.length)

	def test_click(self, mouse, clicktype='add'):
		clicked = False
		if self.scale.contains(mouse):
			clicked = True
			if clicktype == 'add':
				mouse_pos = mouse.getPos()
				self.add_point(self.xpos2point(mouse_pos[0]))
			elif clicktype == 'remove':
				self.remove_point(-1)
		return clicked

	def test_pos(self, mouse):
		if self.scale.contains(mouse):
			mouse_pos = mouse.getPos()
			return self.xpos2point(mouse_pos[0])
		else:
			return None

	def add_point(self, val):
		self.points.append(val)
		pos = [0., self.pos[1]]
		pos[0] = self.point2xpos(val)
		ln = visual.Rect(self.win, pos=pos, width=0.005,
				height=self.h, fillColor=self.line_color,
				lineColor=self.line_color, units=self.units)
		self.lines.append(ln)

	def remove_point(self, ind):
		try:
			self.points.pop(ind)
			self.lines.pop(ind)
		except:
			pass

	def draw(self):
		self.scale.draw()
		[l.draw() for l in self.lines]

	def set_lim(self, newlim):
		self.points = [];
		self.lines = [];
		self.lims = newlim
		self.length = newlim[1] - newlim[0]
