# -*- coding: utf-8 -*-
from __future__ import absolute_import

# add description
# Orientation discrimination task where contrast of gabors is adapted
# according to QUEST procedure.


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
from psychopy.data import QuestHandler, StairHandler

import os
from os import path as op
import pickle
from random import sample

import numpy  as np
import pandas as pd
from PIL import Image

# hackish, but works both for relative import and when run as a script
if __name__ == '__main__' and __package__ is None:
    import sys
    sys.path.append(op.dirname(op.dirname(op.abspath(__file__))))
    __package__ = "GabCon"
    import GabCon # maybe not necessary

from .exputils  import (plot_Feedback, create_database,
                        ContrastInterface, DataManager,
                        ExperimenterInfo, AnyQuestionsGUI)
from .weibull   import Weibull, QuestPlus, weibull_db
from .utils     import to_percent, trim_df
from .stimutils import (exp, db, stim, present_trial, present_break,
    show_resp_rules, textscreen, present_feedback, present_training,
    give_training_db, Instructions, onflip_work, clear_port)
from .viz import plot_quest_plus

if exp['use trigger']:
    from ctypes import windll


# set logging
dm = DataManager(exp)
exp = dm.update_exp(exp)
exp['numTrials'] = 560 # ugly hack, CHANGE
exp['targetTime']  = [1]
exp['SMI']         = [2]

log_path = dm.give_path('l', file_ending='log')
lg = logging.LogFile(f=log_path, level=logging.WARNING, filemode='w')


# TODO: add eeg baseline (resting-state)!
# TODO: add some more logging?

# create object for updating experimenter about progress
exp_info = ExperimenterInfo(exp, stim, main_text_pos=(0, 0.90),
                            sub_text_pos=(0, 0.78))

# from dB and to dB utility functions:
from_db = lambda x: 10 ** (x / 10.)
to_db = lambda x: 10 * np.log10(x)



# show response rules:
show_resp_rules(exp=exp, text=(u"Zaraz rozpocznie się trening." +
    u"\nPrzygotuj się.\nPamiętaj o pozycji palców na klawiaturze."))

# TRAINING
# --------
if exp['run training']:

    # signal onset of training
    core.wait(0.05)
    onflip_work(exp['port'], code='training')
    core.wait(0.1)
    clear_port(exp['port'])

    # set things up
    slow = exp.copy()
    df_train = list()
    num_training_blocks = len(exp['train slow'])
    current_block = 0

    slow['opacity'] = np.array([1.0, 1.0])
    txt = u'Twoja poprawność: {}\nOsiągnięto wymaganą poprawność.\n'
    addtxt = (u'Szybkość prezentacji bodźców zostaje zwiększona.' +
        u'\nAby przejść dalej naciśnij spację.')

    for s, c in zip(exp['train slow'], exp['train corr']):
        # present current training block until correctness is achieved
        df, current_corr = present_training(exp=slow, slowdown=s, corr=c)
        current_block += 1

        # update experimenter info:
        exp_info.training_info([current_block, num_training_blocks],
                               current_corr)

        # show info for the subject:
        if s == 1:
            addtxt = (u'Koniec treningu.\nAby przejść dalej ' +
                      u'naciśnij spację.')
        now_txt = txt + addtxt
        textscreen(now_txt.format(to_percent(current_corr)))
        show_resp_rules(exp=exp)

        # this could be improved:
        # concatenate training db's (and change indexing)
        if 'df_train' in locals():
            if isinstance(df_train, list):
                df_train = trim_df(df)
            else:
                df_train = pd.concat([df_train, trim_df(df)])
                df_train.index = np.r_[1:df_train.shape[0]+1]
        else:
            df_train = trim_df(df)

    # save training database:
    df_train.to_excel(dm.give_path('a'))


# Contrast fitting - Staricase, Quest+
# ------------------------------------
if exp['run fitting']:

    # send start trigger:
    if exp['use trigger']:
        core.wait(0.05)
        onflip_work(exp['port'], code='fitting')
        core.wait(0.1)
        clear_port(exp['port'])

    # init fitting db
    fitting_db = give_training_db(db, slowdown=1)

    # Staircase
    # ---------
    # we start with staircase to make sure subjects familiarize themselves
    # with adapting contrast regime before the main fitting starts
    max_trials = 25
    current_trial = 1
    staircase = StairHandler(0.8, nTrials=max_trials, nUp=1, nDown=2,
                             nReversals=7, minVal=0.001, maxVal=2,
                             stepSizes=[0.1, 0.1, 0.05, 0.05, 0.025, 0.025],
                             stepType='lin')

    for contrast in staircase:
        exp_info.blok_info(u'procedura schodkowa', [current_trial, max_trials])

        # setup stimulus and present trial
        exp['opacity'] = [contrast, contrast]
        core.wait(0.5) # fixed pre-fix interval
        present_trial(current_trial, db=fitting_db, exp=exp)
        stim['window'].flip()

        # set trial type, get response and inform staircase about it
        fitting_db.loc[current_trial, 'trial_type'] = 'staircase'
        response = fitting_db.loc[current_trial, 'ifcorrect']
        staircase.addResponse(response)
        current_trial += 1

        if current_trial % exp['break after'] == 0:
            save_db = trim_df(fitting_db)
            save_db.to_excel(dm.give_path('b'))

            # remind about the button press mappings
            show_resp_rules(exp=exp)
            stim['window'].flip()

    # QUEST+
    # ------
    # next, after about 25 trials we start main fitting procedure - QUEST+
    # TODO: add prior for lapse and maybe other

    # init quest plus
    start_contrast = staircase._nextIntensity
    stim_params = np.arange(-20, 2.1, 0.333) # -20 dB is 0.01 contrast
    model_threshold = stim_params.copy()
    model_slope = np.arange(2., 10., 0.5) # slope in dB too?
    model_lapse = np.arange(0., 0.11, 0.01)

    target_times = [2, 3]
    # loop across target time and SMI
    for ttime in target_times:
        exp['targetTime']  = [ttime]
        exp['SMI']         = [3 - ttime]

        qp = QuestPlus(stim_params, [model_threshold, model_slope, model_lapse],
                       function=weibull_db)

        contrast = stim_params[np.abs(stim_params - to_db(start_contrast)
                                      ).argmin()]

        # update experimenters view:
        draw_prob = False
        step_until = exp['step until']
        block_name = 'Quest Plus - dopasowywanie kontrastu'
        exp_info.blok_info(block_name, [0, 100])

        # remind about the button press mappings
        show_resp_rules(exp=exp)
        stim['window'].flip()

        for trial in range(80):
            # experimenter view
            if draw_prob:
                stim['centerImage'].draw()
            exp_info.blok_info(block_name, [trial + 1, 100])

            # setup stimulus and present trial
            exp['opacity'] = [from_db(contrast), from_db(contrast)]
            core.wait(0.5) # fixed pre-fix interval
            present_trial(current_trial, db=fitting_db, exp=exp)
            stim['window'].flip()

            # set trial type, get response and inform staircase about it
            fitting_db.loc[current_trial, 'trial_type'] = 'Quest+'
            response = fitting_db.loc[current_trial, 'ifcorrect']
            qp.update(contrast, response)
            contrast = qp.next_contrast()
            current_trial += 1

            if current_trial % 10 == 0: # exp['break after']
                save_db = trim_df(fitting_db)
                save_db.to_excel(dm.give_path('b'))

                # visual feedback on parameters probability
                img_name = op.join(exp['data'], 'quest_plus_panel.png')
                fig = plot_quest_plus(qp)
                fig.savefig(img_name, dpi=120)

                # CHECK - make compatible with dual screens
                # set image and its adequate size
                stim['centerImage'].setImage(img_name)
                img = Image.open(img_name)
                imgsize = np.array(img.size)
                del img
                stim['centerImage'].size = imgsize # np.round(imgsize * resize)
                stim['centerImage'].draw()
                exp_info.win.flip()
                if not draw_prob:
                    draw_prob = True

                # remind about the button press mappings
                show_resp_rules(exp=exp)
                stim['window'].flip()

        # save fitting dataframe
        fitting_db = trim_df(fitting_db)
        fitting_db.to_excel(dm.give_path('b'))

        # saving quest may seem unnecessary - posterior can be reproduced
        # from trials, nevertheless we save the posterior as numpy array
        posterior_filename = dm.give_path('posterior', file_ending='npy')
        np.save(posterior_filename, qp.posterior)


# EXPERIMENT - part c
# -------------------
if exp['run main c']:
    # instructions
    if exp['run instruct']:
        instr.present(stop=16)

    # get weibull params from quest plus:
    params = (qp.param_domain * qp.posterior[:, np.newaxis]).sum(axis=0)
    w = Weibull(kind='weibull_db')
    w.params = params

    # get corr treshold from weibull
    contrast_threshold = w.get_threshold([0.55, 1 - params[-1] - 0.01])
    contrast_steps = from_db(np.linspace(*contrast_threshold, num=6))
    logging.warn('contrast steps: ', contrast_steps)

    # 25 repetitions * 4 angles * 6 steps = 600 trials
    db_c = create_database(exp, combine_with=('opacity',
        contrast_steps), rep=25)
    exp['numTrials'] = len(db_c.index)

    # signal that main proc is about to begin
    if exp['use trigger']:
        core.wait(0.05)
        onflip_work(exp['port'], 'contrast')
        core.wait(0.1)
        clear_port(exp['port'])

    # main loop
    for i in range(1, db_c.shape[0] + 1):
        core.wait(0.5) # pre-fixation time is always the same
        present_trial(i, exp=exp, db=db_c, use_exp=False)
        stim['window'].flip()

        # present break
        if i % exp['break after'] == 0:
            # save data before every break
            temp_db = trim_df(db_c.copy())
            temp_db.to_excel(dm.give_path('c'))
            # break and refresh keyboard mapping
            present_break(i, exp=exp)
            show_resp_rules(exp=exp)
            stim['window'].flip()

        # update experimenter
        exp_info.blok_info(u'główne badanie', [i, exp['numTrials']])
        # - [ ] add some figure with correctness for different contrast steps

    db_c.to_excel(dm.give_path('c'))

# goodbye!
# - [ ] some thanks etc. here!
core.quit()
