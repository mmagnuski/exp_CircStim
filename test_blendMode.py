# test all stuff

#fragFBOtoFramePatched = '''
#    uniform sampler2D texture;
#
#    float rand(vec2 seed){
#        return fract(sin(dot(seed.xy ,vec2(12.9898,78.233))) * 43758.5453);
#    }
#
#    void main() {
#        vec4 textureFrag = texture2D(texture,gl_TexCoord[0].st);
#        gl_FragColor.rgb = textureFrag.rgb;
#    }
#    '''
#from psychopy import _shadersPyglet
#_shadersPyglet.fragFBOtoFrame = fragFBOtoFramePatched


from stim import Instructions, Gabor, stim
from psychopy import visual, event, core

stim['window'].close()
win = visual.Window(monitor='testMonitor', fullscr=True, units='deg', 
    useFBO=True, blendMode='add')


instr = Instructions('instructions.yaml', win=win)
instr.present(stop=2)

#gab = Gabor(win=win, mask='gauss', ori=0, sf=1.5, size=5,
#    contrast=1.5)
#gab2 = Gabor(win=win, mask='gauss', ori=0,
#    sf=1.5, size=5, pos=(-8, 0))
#gabors = [gab, gab2]

gab = visual.PatchStim(win=win, mask='gauss', ori=0, sf=1.5, size=5,
    contrast=1.5)
gab2 = visual.PatchStim(win=win, mask='gauss', ori=0,
    sf=1.5, size=5, pos=(-8, 0))
gabors = [gab, gab2]

while True:
	[g.draw() for g in gabors]
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
