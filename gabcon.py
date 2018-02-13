# -*- coding: utf-8 -*-
from __future__ import absolute_import

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
import time
from os import path as op
import pickle
from random import sample

import numpy  as np
import pandas as pd
from PIL import Image

# hackish, but works both for relative import and when run as a script
# CHANGE: relative import should not be needed...
if __name__ == '__main__' and __package__ is None:
    import sys
    sys.path.append(op.dirname(op.dirname(op.abspath(__file__))))
    __package__ = "GabCon"
    import GabCon # maybe not necessary

from .exputils  import (plot_Feedback, create_database, DataManager,
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


# INSTRUCTIONS
# ------------
if exp['run instruct']:
    instr = Instructions('instructions.yaml')
    instr.present(stop=8)
    show_resp_rules(exp=exp, text=u"Tak wygląda ekran przerwy.")
    instr.present(stop=12)
    # "are there any questions" GUI:
    qst_gui = AnyQuestionsGUI(exp, stim)
    qst_gui.run()
    instr.present(stop=13)


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


# Contrast fitting
# ----------------
# staircase, QuestPlus, then QPThreshold

if exp['run fitting']:
    # some instructions
    if exp['run instruct']:
        instr.present(stop=15)

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
    max_trials = exp['staircase trials']
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

    # init quest plus
    start_contrast = staircase._nextIntensity
    min_step = 1. / 255
    stim_params = to_db(np.arange(min_step, 1. + min_step, min_step))
    model_threshold = np.arange(-20, 3., 1.)
    model_slope = np.logspace(np.log10(0.5), np.log10(18.), num=20)
    model_lapse = np.arange(0., 0.16, 0.01)
    stim_params = np.arange(-20, 2.1, 0.333) # -20 dB is 0.01 contrast
    qp = QuestPlus(stim_params, [model_threshold, model_slope, model_lapse],
                   function=weibull_db)
    min_idx = np.abs(stim_params - to_db(start_contrast)).argmin()
    contrast = stim_params[min_idx]

    # update experimenters view:
    block_name = 'Quest Plus - dopasowywanie kontrastu'
    exp_info.blok_info(block_name, [0, 100])

    # remind about the button press mappings
    show_resp_rules(exp=exp)
    stim['window'].flip()

    for trial in range(exp['QUEST plus trials']):
        # CHECK if blok_info flips the screen, better if not...
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

            # remind about the button press mappings
            show_resp_rules(exp=exp)
            stim['window'].flip()

            # visual feedback on parameters probability
            t0 = time.clock()
            img_name = op.join(exp['data'], 'quest_plus_panel.png')
            plot_quest_plus(qp).savefig(img_name, dpi=120)

            exp_info.experimenter_plot(img_name)
            time_delta = time.clock() - t0
            msg = 'time taken to update QuestPlus panel plot: {:.3f}'
            logging.warn(msg.format(time_delta))

    # saving quest may seem unnecessary - posterior can be reproduced
    # from trials, nevertheless it is useful for debugging
    posterior_filename = dm.give_path('posterior', file_ending='npy')
    np.save(posterior_filename, qp.posterior)

    # threshold fitting
    # -----------------

    # initialize further threshold optimization
    trimmed_df = trim_df(fitting_db)
    qps, corrs = init_thresh_optim(trimmed_df, qp)
    block_name = 'threshold optimization'

    t0 = time.clock()
    ax = plot_threshold_entropy(qps, corrs=corrs)
    img_name = op.join(exp['data'], 'quest_plus_thresholds.png')
    ax.figure.savefig(img_name, dpi=180)
    exp_info.experimenter_plot(img_name)
    time_delta = time.clock() - t0
    msg = 'time taken to update QuestPlus threshold plot: {:.3f}'
    logging.warn(msg.format(time_delta))

    # optimize thresh...
    for trial in range(exp['threshold opt trials']):
        # select contrast
        posteriors = [qp.get_threshold().sum(axis=(1, 2)) for qp in qps]
        posterior_peak = [posterior.max() for posterior in posteriors]
        optimize_threshold = np.array(posterior_peak).argmin()
        contrast = qps[optimize_threshold].next_contrast(axis=(1, 2))

        # CHECK if blok_info flips the screen, better if not...
        exp_info.blok_info(block_name, [trial + 1, 50])

        # setup stimulus and present trial
        exp['opacity'] = [contrast, contrast]
        core.wait(0.5) # fixed pre-fix interval
        present_trial(current_trial, db=fitting_db, exp=exp)
        stim['window'].flip()

        # set trial type, get response and inform staircase about it
        fitting_db.loc[current_trial, 'trial_type'] = 'Quest+ thresholds'
        response = fitting_db.loc[current_trial, 'ifcorrect']
        for qp in qps:
            qp.update(contrast, response)
        current_trial += 1

        if current_trial % 10 == 0: # exp['break after']
            save_db = trim_df(fitting_db)
            save_db.to_excel(dm.give_path('b'))

            # remind about the button press mappings
            show_resp_rules(exp=exp)
            stim['window'].flip()

            # visual feedback on parameters probability
            t0 = time.clock()
            ax = plot_threshold_entropy(qps, corrs=corrs)
            ax.figure.savefig(img_name, dpi=180)
            exp_info.experimenter_plot(img_name)
            time_delta = time.clock() - t0
            msg = 'time taken to update QuestPlus threshold plot: {:.3f}'
            logging.warn(msg.format(time_delta))

    # save fitting dataframe
    fitting_db = trim_df(fitting_db)
    fitting_db.to_excel(dm.give_path('b'))

    # saving quest may seem unnecessary - posterior can be reproduced
    # from trials, nevertheless we save the posterior as numpy array
    posterior_filename = dm.give_path('posterior_thresh', file_ending='npy')
    np.save(posterior_filename, qps[2].posterior)
    # + save stim space


# EXPERIMENT - part c
# -------------------
if exp['run main c']:
    # instructions
    if exp['run instruct']:
        instr.present(stop=16)

    # get contrast thresholds from quest plus:
    contrasts = list()
    for idx, qp in enumerate(qps):
        wb_args = dict(kind='weibull', corr_at_thresh=corrs[idx])
        params = qp.get_fit_params(select='ML', weibull_args=wb_args)
        contrasts.append(params[0])

    logging.warn('final contrast steps: ', contrasts)

    # 30 repetitions * 4 angles * 5 steps = 600 trials
    db_c = create_database(exp, combine_with=('opacity', contrasts), rep=30)
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
