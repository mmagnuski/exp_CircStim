import os
import os.path as op
import random
import warnings
from ctypes import windll

from psychopy import visual, event, monitors, core, sound
from stimutils import Instructions


def send_trigger(portdict, code):
    if portdict['send']:
		windll.inpout32.Out32(portdict['port address'], code)

def run(window, exp, segment_time=60., debug=False, instr_dir='instr',
        exp_info=None):
    segment_time = 1. if debug else segment_time

    # present instructions
    # check participant sex
    sex = exp['participant']['sex'][0]
    img_files = [op.join(instr_dir, fl) for fl in os.listdir(instr_dir)
                 if fl.startswith(sex + '_') and fl.lower().endswith('.png')]
    instr = Instructions(img_files, images=True, auto=exp['debug'],
                         image_args=dict(size=[1451, 816], units='pix'),
                         exp_info=exp_info)
    instr.present()
    window.flip()

    # choose seqence:
    possible_seq = ['OCCOCOOC', 'COOCOCCO']
    seq = random.sample(possible_seq, 1)[0]

    sound_files = {s[0].upper(): op.join('sound', s)
                   for s in ['open.wav', 'close.wav']}
    trig ={'O': 10, 'C': 11}
    stop_sound = sound.Sound(os.path.join('sound', 'stop.wav'))
    sounds = {k: sound.Sound(sound_files[k]) for k in sound_files.keys()}

    for s in seq:
        # check for quit
        resp = event.getKeys()
        if 'q' in resp:
            core.quit()

        # play open/close sound
        snd = sounds[s]
        snd.play()

        # set trigger
        send_trigger(exp['port'], trig[s])
        window.flip()
        core.wait(0.1)
        send_trigger(exp['port'], 0)

        # wait segment_time, play ring and then wait break time
        core.wait(segment_time)
        stop_sound.play()
        wait_time = 0.5 if debug else random.random() * 3 + 3.5
        core.wait(wait_time)

    return window
