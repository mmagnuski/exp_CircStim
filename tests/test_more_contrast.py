# monkey patching shaders
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

# classical imports
from psychopy import visual, event, core

class Gabor(object):
	def __init__(self, **kwargs):
		self.draw_which = 0
		win = kwargs.pop('win')
		self.win = win
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
		# make sure window is in blendMode 'add':
		if self.win.blendMode == 'add':
			self.win.blendMode = 'add'
		# draw all gabors that need to be drawn:
		for g in range(self.draw_which):
			self.gabors[g].draw()


win = visual.Window(monitor='testMonitor', units='deg', 
    useFBO=True, blendMode='add')
gab = Gabor(win=win, mask='gauss', ori=0, sf=1.5, size=5,
    contrast=1.5)
gab2 = Gabor(win=win, mask='gauss', ori=0,
    sf=1.5, size=5, pos=(-8, 0))
text = visual.TextStim(win, pos=(0,6), text='Hello!')
gabors = [gab, gab2]

steps = [-0.01, 0.01]
while True:
	for g in gabors:
		g.draw()
	text.draw()
	win.flip()
	k = event.getKeys()
	if k:
		print k
	if 'q' in k:
		core.quit()
	elif 'up' in k:
		gab.set_contrast(gab.contrast + 0.1)
	elif 'down' in k:
		gab.set_contrast(gab.contrast - 0.1)
	core.wait(0.01)

