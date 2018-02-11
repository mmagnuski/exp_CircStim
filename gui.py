from psychopy import visual
from exputils import Interface

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


class ContrastInterface(Interface):
	def __init__(self, exp=None, stim=None, contrast_lims=(0., 3.),
		df=None, weibull=None, timeout=False, num_trials=None,
		set_image_size=False, contrast_method='4steps',
		corr_steps=(0.55, 0.75, 0.95)):
		from weibull import fitw, get_new_contrast

		self.contrast = list()
		self.get_new_contrast = get_new_contrast

		# monitor setup
		# -------------
		self.wait_text = u'Proszę czekać, trwa dobieranie kontrastu...'
		super(ContrastInterface, self).__init__(exp, stim)

		self.win.setMouseVisible(True)
		self.mouse = event.Mouse(win=self.win)

		# change units, size and position of centerImage:
		self.origunits = self.win.units
		self.win.units = 'norm'
		self.set_image_size = set_image_size

		if df is not None:
			self.df = trim_df(df.reset_index())
			nrow = self.df.shape[0]
		else:
			self.df = df
			nrow = 10

		# weibull checks
		use_lapse = False
		if weibull is not None:
			self.weibull = weibull
			self.params = weibull.params
			use_lapse = len(weibull.params) == 3
			if num_trials is None:
				num_trials = len(weibull.y)
		else:
			self.weibull = None
			self.params = None

		self.fitfun = fitw
		self.use_lapse = use_lapse
		self.num_trials = num_trials if num_trials is not None else 40
		self.num_trials = max(5, min(nrow, self.num_trials))
		self.corr_steps = corr_steps

		win_pix_size = self.win.size
		pic_pix_size = self.stim['centerImage'].size
		pic_nrm_size = [(p / (w * 1.)) * 2. for (p, w) in
			zip(pic_pix_size, win_pix_size)]
		self.stim['centerImage'].units = 'norm'
		self.stim['centerImage'].setPos((-0.4, 0.4))
		self.stim['centerImage'].setSize([pic_nrm_size])

		# buttons
		button_pos = np.zeros([4,2])
		button_pos[:,0] = 0.7
		button_pos[:,1] = np.linspace(-0.25, -0.8, num=4)
		button_text = [u'kontynuuj', u'zakończ', u'edytuj', '0.1']
		self.buttons = [Button(win=self.win, pos=p, text=t,
			size=(0.35, 0.12)) for p, t in zip(button_pos, button_text)]
		self.buttons[-1].click_fun = self.cycle_vals


		txt = ' '.join([str(self.num_trials), '/', str(nrow), 'trials'])
		self.last_trials = Button(win=self.win, pos=(-0.7, -0.43),
			text=txt, size=(0.4, 0.12))
		self.last_trials.click_fun = self.set_last_trials

		self.use_lapse_button = Button(win=self.win, pos=(-0.7, -0.6166),
			text='use lapse',size=(0.4, 0.12))
		self.use_lapse_button.click_fun = self.change_lapse
		if self.use_lapse:
			self.use_lapse_button.default_click_fun()
		self.contrast_method = contrast_method
		self.current_method_string = contrast_method[:-5] if contrast_method \
			is not None else ''
		self.contrast_method_button = Button(win=self.win, pos=(-0.7, -0.803),
			text='steps: ' + self.current_method_string, size=(0.4, 0.12))
		self.contrast_method_button.click_fun = self.change_contrast_method
		self.weibull_contrast_steps = None

		# text
		self.text = visual.TextStim(win=self.win, text='kontrast:\n',
			pos=(0.75, 0.5), units='norm', height=0.1)
		self.text_height = {0: 0.12, 5: 0.1, 9: 0.06, 15: 0.04, 20: 0.03}
		self.scale_text = visual.TextStim(self.win, pos=(0.,-0.65), text="")
		self.contrast_levels_text = visual.TextStim(win=self.win, text='',
			pos=(0.3, 0.5), height=0.06)

		# scale
		self.scale = ClickScale(win=self.win, pos=(0.,-0.8), size=(0.75, 0.1),
								lims=contrast_lims)
		self.scale2 = ClickScale(win=self.win, pos=(0.,-0.9), size=(0.75, 0.05),
								 lims=contrast_lims)
		self.edit_mode = False
		self.last_pressed = False
		self.orig_contrast_lims = contrast_lims
		self.current_contrast_lims = contrast_lims
		self.grain_vals = [0.1, 0.05, 0.01, 0.005]
		self.current_grain_val = 0
		self.in_loop2 = False
		self.in_method_loop = False

		# back in how many trials:
		self.next_trials = 4
		self.trials_text = visual.TextStim(win=self.win,
			text=u'wróć po {} trialach'.format(self.next_trials),
			pos=(0., -0.45), units='norm', height=0.1)

		# timeout
		self.in_timeout = True if timeout else False
		if self.in_timeout:
			self.timeout = timeout
			self.timeout_stopped = False
			self.timeout_val = int(round(timeout))
			self.timeout_text = visual.TextStim(win=self.win, units='norm',
				text=str(self.timeout_val), color=(1, -1, -1), height=1.5)


	def draw(self):
		[b.draw() for b in self.buttons]
		self.last_trials.draw()
		self.use_lapse_button.draw()
		self.trials_text.draw()
		self.contrast_levels_text.draw()
		self.contrast_method_button.draw()
		self.stim['centerImage'].draw()

		if self.in_timeout:
			self.timeout_text.draw()

		# edit constrast mode
		if self.buttons[-2].clicked:
			self.set_scale_text()
			self.scale.draw()
			self.scale2.draw()
			self.scale_text.draw()
			# TODO: this might be moved to refresh:
			if self.last_pressed:
				step = self.grain_vals[self.current_grain_val]
				self.contrast = round2step(np.array(self.scale.points),
										   step=step)
				txt = 'kontrast:\n' + '\n'.join(map(str, self.contrast))
				num_cntrst = len(self.contrast)
				k = np.sort(self.text_height.keys())
				sel = np.where(np.array(k) <= num_cntrst)[0][-1]
				self.text.setHeight(self.text_height[k[sel]])
				self.text.setText(txt)
				self.last_pressed = False
			self.text.draw()

	def set_last_trials(self):
		self.in_loop2 = True
		self.last_trials.default_click_fun()
		self.loop2()

	def change_contrast_method(self):
		self.in_method_loop = True
		self.contrast_method_button.default_click_fun()
		self.method_loop()

	def check_contrast_method(self, method):
		if not isinstance(method, str):
			raise ValueError('method must be a string.')
		if method == '':
			return None

		default = '4steps'
		num = ''.join([c for c in method if c.isdigit()])
		if len(num) < 1:
			return default
		if int(num) > 25:
			return default
		rest = ''.join([c for c in method if not c.isdigit()])
		if rest not in ['log', 'midlog']:
			return num + 'steps'
		else:
			return num + rest + 'steps'

	def change_lapse(self):
		self.use_lapse_button.default_click_fun()
		self.use_lapse = self.use_lapse_button.clicked
		self.refresh_weibull()

	def cycle_vals(self):
		self.current_grain_val += 1
		if self.current_grain_val >= len(self.grain_vals):
			self.current_grain_val = 0
		self.buttons[-1].setText(str(self.grain_vals[self.current_grain_val]))
		# TODO: change contrast values?

	def refresh(self):
		self.check_key_press()
		if self.in_timeout:
			time = self.countdown.getTime()
			if time > 0:
				round_time = int(round(time))
				if round_time < self.timeout_val:
					self.timeout_val = round_time
					self.timeout_text.setText(str(self.timeout_val))
					self.draw()
					self.win.flip()
			else:
				self.runLoop = False

		else:
			if_click = self.check_mouse_click()
			self.last_pressed = if_click
			if not self.edit_mode and self.buttons[-1].clicked:
				self.edit_mode = True
			self.draw()
			self.win.flip()
			if if_click:
				core.wait(0.1)


	def set_scale_text(self):
		for obj in [self.scale, self.scale2]:
			val = obj.test_pos(self.mouse)
			if val:
				step = self.grain_vals[self.current_grain_val]
				val = round2step(val, step=step)
				get_chars = 2 + np.sum(np.array([1., 0.1, 0.01, 0.001]) <= val)
				val = str(val)
				get_chars = min(get_chars, len(val))
				val = val[:get_chars]
				break
			else:
				val = ""
		self.scale_text.setText(val)

	def check_scale2(self):
		vals = sorted(self.scale2.points)
		if vals:
			step = self.grain_vals[self.current_grain_val]
			vals = round2step(np.asarray(vals), step=step)
			if len(vals) == 1:
				self.current_contrast_lims = (0., vals[0])
			elif len(vals) >= 2:
				self.current_contrast_lims = (vals[0], vals[-1])
		else:
			self.current_contrast_lims = self.orig_contrast_lims
		self.scale.set_lim(self.current_contrast_lims)

	def check_mouse_click(self):
		m1, m2, m3 = self.mouse.getPressed()
		if m1:
			self.mouse.clickReset()
			# test buttons
			ifclicked = [b.contains(self.mouse) for b in self.buttons]
			which_clicked = np.where(ifclicked)[0]
			if which_clicked.size > 0:
				self.buttons[which_clicked[0]].click()

			# test scale
			self.scale.test_click(self.mouse)
			if_clicked = self.scale2.test_click(self.mouse)
			if if_clicked:
				self.check_scale2()

			# test trial num edit
			if self.last_trials.contains(self.mouse):
				self.last_trials.click()

			if self.use_lapse_button.contains(self.mouse):
				self.use_lapse_button.click()

			if self.contrast_method_button.contains(self.mouse):
				self.contrast_method_button.click()

		elif m3:
			self.mouse.clickReset()
			self.scale.test_click(self.mouse, 'remove')
			if_clicked = self.scale2.test_click(self.mouse, 'remove')
			if if_clicked:
				self.check_scale2()
		return m1 or m3

	def check_key_press(self):
		k = event.getKeys()
		if k:
			update_next_trials = False
			if 'minus' in k:
				self.next_trials -= 1
				self.next_trials = max(3, self.next_trials)
				update_next_trials = True
			if 'equal' in k:
				self.next_trials += 1
				update_next_trials = True
			if self.in_timeout:
				if np.any([x in k for x in ['return', 'space']]):
					self.in_timeout = False
					self.timeout_stopped = True
			else:
				if 'q' in k or 'return' in k:
					self.runLoop = False
			if update_next_trials:
				self.trials_text.setText(u'wróć po {} trialach'.format(
					self.next_trials))

	def refresh_weibull(self):
		# fit weibull
		nrow = self.df.shape[0]
		look_back = self.num_trials
		ind = np.r_[nrow-look_back:nrow]
		params = [1., 1., 0.] if self.use_lapse else [1., 1.]
		self.weibull = self.fitfun(self.df, ind, init_params=params)
		self.params = self.weibull.params

		# update contrast for specified correctness
		if self.contrast_method is not None:
			if len(np.unique(self.contrast)) == 2:
				contrast, _ = self.get_new_contrast(self.weibull,
					contrast_lims=np.unique(self.contrast),
					method=self.contrast_method)
			else:
				contrast, _ = self.get_new_contrast(self.weibull,
					corr_lims=self.exp['fitCorrLims'],
					method=self.contrast_method)
			self.weibull_contrast_steps = contrast
		else:
			if len(self.contrast) > 0:
				self.weibull_contrast_steps = np.unique(self.contrast)
			else:
				self.weibull_contrast_steps = None

		plotter_args = dict(mean_points=True, min_bucket=0.01,
							line_color='seaborn_red')
		if self.weibull_contrast_steps is not None:
			plotter_args.update(dict(contrast_steps=self.weibull_contrast_steps))

		self.stim = plot_Feedback(self.stim, self.weibull, self.exp['data'],
			plotter_args=plotter_args, set_size=self.set_image_size)
		if self.set_image_size:
			self.set_image_size = False
		self.stim['centerImage'].draw()



		# CHANGE - think a little more on when to leave the default - corr range -> contrast
		if self.weibull_contrast_steps is None:
			corr_steps = self.corr_steps
			contrast = self.weibull.get_threshold(self.corr_steps)
		else:
			contrast = self.weibull_contrast_steps
			corr_steps = self.weibull.predict(contrast)
		txt = ['{}% - {:05.3f}'.format(str(corr * 100).split('.')[0], cntr)
			for corr, cntr in zip(corr_steps, contrast)]
		txt = ';\n'.join(txt)
		self.contrast_levels_text.setText(txt)

	def test_keys_loop2(self, k):
		if k and self.in_loop2:
			k = k[0]
			current_str = str(self.num_trials)
			if current_str == '0':
				current_str = ''
			# backspace - remove char from num_trials
			if k == 'backspace':
				if len(current_str) > 0:
					current_str = current_str[:-1]

			# number - add to num_trials
			if k in list('1234567890'):
				current_str += k

			nrow = self.df.shape[0]
			self.num_trials = int(current_str) if \
				len(current_str) > 0 else 0
			txt = str(self.num_trials) if self.num_trials > 0 else ''
			txt = ' '.join([txt, '/', str(nrow), 'trials'])
			self.last_trials.setText(txt)

			# return - refresh weibull
			if k == 'return':
				self.num_trials = max(5, min(nrow, self.num_trials))
				txt = ' '.join([str(self.num_trials), '/', str(nrow), 'trials'])
				self.last_trials.setText(txt)
				self.refresh_weibull()
				self.in_loop2 = False
				self.last_trials.default_click_fun()

	def test_keys_loop_method(self, k):
		if k and self.in_method_loop:
			k = k[0]
			current_str = self.current_method_string
			# backspace - remove char from num_trials
			if k == 'backspace':
				if len(current_str) > 0:
					current_str = current_str[:-1]

			# number - add to num_trials
			if k in list('1234567890midlog'):
				current_str += k

			self.current_method_string = current_str
			txt = ': '.join(['steps', current_str])
			self.contrast_method_button.setText(txt)
			# return - refresh weibull
			if k == 'return':
				last_contrast_method = self.contrast_method
				self.contrast_method = self.check_contrast_method(current_str)
				self.current_method_string = self.contrast_method[:-5] if \
					self.contrast_method is not None else ''
				self.contrast_method_button.setText('steps: ' + \
					self.current_method_string)
				self.in_method_loop = False
				if not self.contrast_method == last_contrast_method:
					self.refresh_weibull()
				self.contrast_method_button.default_click_fun()

	def loop2(self):
		while self.in_loop2:
			k = event.getKeys(
				keyList=list('1234567890') + ['return', 'backspace'])
			self.test_keys_loop2(k)
			self.draw()
			self.win.flip()

	def method_loop(self):
		while self.in_method_loop:
			k = event.getKeys(keyList=list('1234567890midlog') + \
				['return', 'backspace'])
			self.test_keys_loop_method(k)
			self.draw()
			self.win.flip()

	def quit(self):
		self.win.setMouseVisible(False)
		self.win.units = self.origunits

	def loop(self):
		self.runLoop = True
		continue_fitting = True
		if self.in_timeout:
			self.countdown = core.CountdownTimer(self.timeout)
		while self.runLoop:
			self.refresh()
			if self.buttons[1].clicked:
				self.runLoop = False
				continue_fitting = False
			elif self.buttons[0].clicked:
				self.runLoop = False
				continue_fitting = True
		self.quit()
		return continue_fitting
