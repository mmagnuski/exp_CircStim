# -*- coding: utf-8 -*-

#fragFBOtoFramePatched = '''
#   uniform sampler2D texture;
#
#   float rand(vec2 seed){
#       return fract(sin(dot(seed.xy ,vec2(12.9898,78.233))) * 43758.5453);
#   }
#
#   void main() {
#       vec4 textureFrag = texture2D(texture,gl_TexCoord[0].st);
#       gl_FragColor.rgb = textureFrag.rgb;
#   }
#   '''
#from psychopy import _shadersPyglet
#_shadersPyglet.fragFBOtoFrame = fragFBOtoFramePatched

from psychopy import visual, event, core

win = visual.Window(monitor='testMonitor', useFBO=True,
    blendMode='add', units='deg')

g = visual.GratingStim(win, tex='sin', mask='gauss', size=4.5, pos=(0,6))
t = visual.TextStim(win=win, text='blendMode={}'.format(win.blendMode))

draw_order = [[g], [g, t], [g, t]]
for draw_now in draw_order:
    for stim in draw_now:
        stim.draw()
    win.flip()
    win.blendMode = 'add'
    event.waitKeys()

core.quit()
