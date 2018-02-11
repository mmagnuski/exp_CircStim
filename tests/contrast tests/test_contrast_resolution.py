# -*- coding: utf-8 -*-
from __future__ import absolute_import

# Procedure for choosing contrast steps

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

from psychopy import __version__ as psychopy_version
from distutils.version import LooseVersion

psychopy_version = LooseVersion(psychopy_version)

if psychopy_version >= LooseVersion('1.84'):
    from psychopy.visual import shaders
else:
    from psychopy import _shadersPyglet as shaders

shaders.fragFBOtoFrame = fragFBOtoFramePatched

# other imports
# -------------
from psychopy  import visual, core, event, logging

import os
from os import path as op

import numpy  as np
from PIL import Image
from stimutils import stim


def gaussian(size, sigma=1.):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel


select_contrasts = False
required_threshold = 450.

orientation = 45
target = stim['target'][orientation]
contrasts = np.arange(0.000, 1.5005, step=0.0001)
window = stim['window']

previous_frame = None
diffs = list()
accepted_contrasts = list()

# find frame where gabor is located
target.set_contrast(2.0)
target.draw()
window.flip()

current_frame = np.array(window.getMovieFrame())[..., 0].astype('float')
whr = np.where(~(current_frame == 128))
funs = [lambda x: min(x) - 2, lambda x: max(x) + 3]
i1, i2 = [f(whr[0]) for f in funs]
j1, j2 = [f(whr[1]) for f in funs]

n_pix = i2 - i1
gauss = gaussian(n_pix, sigma=int(n_pix/5))


for cnt_idx, cnt in enumerate(contrasts):
    target.set_contrast(cnt)
    target.draw()
    window.flip()

    # get print screen
    current_frame = np.array(window.getMovieFrame())[..., 0].astype('float')
    current_frame = current_frame[i1:i2, j1:j2]
    window.movieFrames = list()

    if previous_frame is not None:
        frame_diff = (np.abs(current_frame - previous_frame) * gauss).sum()
        diffs.append(frame_diff)

    # compare to previous
    if select_contrasts:
        if frame_diff >= required_threshold:
            accepted_contrasts.append(cnt)
            diffs.append(frame_diff)
            previous_frame = current_frame
        else:
            # remove file from disk
            os.remove(fname)
    previous_frame = current_frame

    if cnt_idx % 10 == 0:
        print('{} / {}'.format(cnt_idx, len(contrasts)))
        try:
            print('last diff: {}'.format(frame_diff))
        except:
            pass

# save accepted contrast values:
if select_contrasts:
    with open('contrasts.txt', 'w') as f:
        for cnt in accepted_contrasts:
            f.write('{:.4f}'.format(cnt) + '\n')

# save accepted contrast values:
with open('diffs.txt', 'w') as f:
    for diff in diffs:
        f.write('{:.1f}'.format(diff) + '\n')


core.quit()
