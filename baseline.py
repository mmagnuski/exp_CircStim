import os
import random
import warnings
from ctypes import windll

from psychopy import visual, event, monitors, core, sound


def send_trigger(exp, code):
    if exp['use trigger']:
		windll.inpout32.Out32(exp['port address'], code)

def run(window, segment_time=60., debug=False, instr_dir='instr'):
    segment_time = 0.5 if debug else segment_time

    # present instructions
    img_dir = os.path.join(instr_dir, 'baseline.png')
    img = visual.ImageStim(window, image=img_dir, size=[1169, 826],
                           units='pix', interpolate=True)
    img.draw(); window.flip()
    event.waitKeys(keyList=['right'])
    window.flip()

    # choose seqence:
    possible_seq = ['OCCOCOOC', 'COOCOCCO']
    seq = random.sample(possible_seq, 1)[0]

    sound_files = {s[0].upper(): op.join('sound', s)
                   for s in ['open.wav', 'close.wav']]
    trig ={'O': 10, 'C': 11}
    stop_sound = sound.Sound(os.path.join('snd', 'stop.wav'))
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
        send_trigger(settings, trig[s])
        window.flip()
        core.wait(0.1)
        send_trigger(settings, 0)

        # wait segment_time, play ring and then wait break time
        core.wait(segment_time)
        stop_sound.play()
        wait_time = 0.5 if debug else random.random() * 3 + 3.5
        core.wait(wait_time)

    return window
