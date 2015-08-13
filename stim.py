# -*- coding: utf-8 -*-

# monkey-patch pyglet shaders:
# ----------------------------
fragFBOtoFramePatched = '''
    uniform sampler2D texture;

    float rand(vec2 seed){
        return fract(sin(dot(seed.xy ,vec2(12.9898,78.233))) * 43758.5453);
    }

    void main() {
        vec4 textureFrag = texture2D(texture,gl_TexCoord[0].st);
        gl_FragColor.rgb = textureFrag.rgb;
    }
    '''
from psychopy import _shadersPyglet
_shadersPyglet.fragFBOtoFrame = fragFBOtoFramePatched

# imports
# -------
from psychopy import visual, event
from random    import randint
import yaml
import re


# create a window
# ---------------
win = visual.Window(monitor='testMonitor', fullscr=True, units='deg', 
    useFBO=True, blendMode='add')

# prepare stimuli
# ---------------
stim = {}
stim['window'] = win

# win.setMouseVisible(False)

exp = dict()

exp['orientation'] = [0, 45, 90, 135]
exp['use keys']    = ['f', 'j']
exp['participant']  = dict()
exp['participant']['sex'] = u'mężczyzna'
exp['gabor size']  = 5
exp['gabor freq']  = 1.5
resp = {}
choose_resp = randint(0, 1)
resp[0]   = exp['use keys'][choose_resp]
resp[90]  = exp['use keys'][choose_resp]
resp[45]  = exp['use keys'][1 - choose_resp]
resp[135] = exp['use keys'][1 - choose_resp]
exp['choose_resp'] = choose_resp
exp['keymap'] = resp

# STIMULI
# -------

def txt(win=win, **kwargs):
	return visual.TextStim(win, units='norm', **kwargs)


def txt_newlines(win=win, exp=exp, text='', **kwargs):
	text = text.replace('\\n', '\n')
	text = text.replace('[90button]', exp['keymap'][90])
	text = text.replace('[45button]', exp['keymap'][45])

	# check for gender related formulations
	ptrn = re.compile(r'\[(\w+)/(\w+)\]', flags=re.U)
	found = ptrn.finditer(text)
	new_text = ''; last_ind = 0
	for f in found:
		ind = f.span()
		grp = f.groups()
		correct_text = grp[exp['participant']['sex'] == 'k']
		new_text += text[last_ind:ind[0]] + correct_text
		last_ind = ind[1]
	if new_text:
		new_text += text[last_ind:]
		text = new_text

	return visual.TextStim(win, text=text, units='norm', **kwargs)


# gabor creation
def gabor(win = win, ori = 0, opa = 1.0,
		  pos  = [0, 0], size = exp['gabor size'],
		  units = 'deg', sf = exp['gabor freq']):
	return visual.PatchStim(win     = win,  mask = "gauss", \
							  size    = size, pos  = pos, \
							  sf      = sf,   ori  = ori,      \
							  opacity = opa,  units = units)


class Gabor(object):
	'''Simple gabor class that allows the contrast to go
	up to 3.0 (clipping the out of bound rgb values).
	Requires window to be set with useFBO=True and 
	blendMode='add' as well as a monkey-patch for
	pyglet shaders.'''

	def __init__(self, **kwargs):
		self.draw_which = 0
		win = kwargs.pop('win')
		self.contrast = kwargs.pop('contrast', 1.)
		kwargs.update({'contrast': 0.})

		# generate gabors:
		self.gabors = list()
		self.gabors.append(visual.GratingStim(win, **kwargs))
		self.gabors.append(visual.GratingStim(win, **kwargs))
		self.gabors.append(visual.GratingStim(win, **kwargs))
		self.set_contrast(self.contrast)

	def set_contrast(self, val):
		self.contrast = val
		self.draw_which = 0
		for g in self.gabors:
			if val > 0.:
				self.draw_which += 1
				if val > 1.:
					g.contrast = 1.
					val = val - 1.
				else:
					g.contrast = val
					break

	def draw(self):
		for g in range(self.draw_which):
			self.gabors[g].draw()


def fix(win=stim['window'], color=(0.5, 0.5, 0.5)):
	dot = visual.Circle(win, radius=0.15,
		edges=16, units='deg')
	dot.setFillColor(color)
	dot.setLineColor(color)
	return dot


def feedback_circle(win=win, radius=2.5, edges=64,
	color='green', pos=[0,0]):
	color_mapping = {'green': [0.1, 0.9, 0.1], 'red': [0.9, 0.1, 0.1]}
	color = color_mapping[color]
	circ = visual.Circle(win, pos=pos, radius=radius, edges=edges, units='deg')
	circ.setFillColor(color)
	circ.setLineColor(color)
	return circ





class Instructions:

	nextpage   = 0
	mapdict    = {'gabor': gabor, 'text': txt_newlines,
		'fix':fix, 'feedback': feedback_circle}
	navigation = {'left': 'prev',
				  'right': 'next',
				  'space': 'next'}

	def __init__(self, fname, win=win):
		self.win = win

		# get instructions from file:
		with open(fname, 'r') as f:
		    instr = yaml.load_all(f)
		    self.pages = list(instr)
		self.stop_at_page = len(self.pages)

	def present(self, start=None, stop=None):
		if not isinstance(start, int):
			start = self.nextpage
		if not isinstance(stop, int):
			stop = len(self.pages)

		# show pages:
		self.nextpage = start
		while self.nextpage < stop:
			# create page elements
			self.create_page()
			# draw page elements
			for it in self.pageitems:
				it.draw()
			self.win.flip()

			# wait for response
			k = event.waitKeys(keyList=self.navigation.keys())[0]
			action = self.navigation[k]

			# go next/prev according to the response
			if action == 'next':
				self.nextpage += 1
			else:
				self.nextpage = max(0, self.nextpage - 1)

	def create_page(self, page_num=None):
		if not isinstance(page_num, int):
			page_num = self.nextpage
		self.pageitems = [self.parse_item(i) for i in self.pages[page_num]]

	def parse_item(self, item):
		# currently: gabor or text:
		fun = self.mapdict.get(item['item'], [])
		args = item['value']
		args.update({'win' : self.win})
		if fun: return fun(**item['value'])
