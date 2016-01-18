# gets fitting_db

# FinalFitGUI
# 
# Input args:
# * fitting_db
# * weibull instance
# * exp and window
#
# - handles two screens
# - button for ok
# - edit text (accept enter) for num last trials to fit
# - refresh fit
# - starting look_back = 40


if not 'window2' in stim:
	stim['window'].blendMode = 'avg'



stim = plot_Feedback(stim, w, exp['data'])
interf = ContrastInterface(stim=stim, trial=trial)


class FinalFitGUI(Interface):

	def __init__(self, exp=None, stim=None, db=None, weibull=None):
		self.contrast = []

		# setup
		# -----
		self.wait_text = u'Proszę czekać, trwa dobieranie kontrastu...'	
		super(ContrastInterface, self).__init__(exp, stim)	

		self.win.setMouseVisible(True)
		self.mouse = event.Mouse(win=self.win)
		self.db = db
		self.weibull = weibull

		# centerImage (weibull fit plot):
		self.origunits = self.win.units
		self.win.units = 'norm'
		win_pix_size = self.win.size
		pic_pix_size = self.stim['centerImage'].size
		pic_nrm_size = [(p / (w * 1.)) * 2. for (p, w) in
			zip(pic_pix_size, win_pix_size)]
		self.stim['centerImage'].units = 'norm'
		self.stim['centerImage'].setPos((0., 0.2))
		self.stim['centerImage'].setSize([pic_nrm_size])

		# OK button
		pos = [0., -0.8]
		txt = ['OK']
		self.OKbutton = Button(win=self.win, pos=pos, text=txt, size=(0.35, 0.12))
		self.OKbutton.click_fun = self.accept
		self.notfinished = True
		
		# edit box
		self.num_trials = 40
		self.text = visual.TextStim(win=self.win, text=str(self.num_trials),
			pos=(0.0, -0.5), units='norm', height=0.16)

	def draw(self):
		self.OKbutton.draw()
		self.text.draw()
		self.stim['centerImage'].draw()

	def refresh_weibull(self):

		
	def test_keys(self, k):
		if k:
			current_str = str(self.num_trials)
			# backspace - remove char from num_trials
			if k == 'backspace':
				if len(current_str) > 0:
					current_str = current_str[:-1]

			# number - add to num_trials
			if k in list('1234567890'):
				current_str += k

			self.text.setText(current_str)
			self.num_trials = int(current_str)

			# return - refresh weibull
			if k == 'return':
				self.refresh_weibull()

	def GUI_loop(self):
		while self.notfinished:
			k = event.getKeys(keyList=list('1234567890')+['return', 'backspace'])
			self.test_keys(k)

			self.draw()
			self.win.flip()

	def check_mouse_click(self):
		m1, m2, m3 = self.mouse.getPressed()
		if m1:
			self.mouse.clickReset()
			# test buttons
			if self.OKbutton.contains(self.mouse):
				self.OKbutton.click()

	def accept(self):
		self.notfinished = False
