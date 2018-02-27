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
import matplotlib.pyplot as plt
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
from .weibull   import (Weibull, QuestPlus, weibull_db, PsychometricMonkey,
                        init_thresh_optim, plot_threshold_entropy, from_db,
                        to_db, weibull)
from .utils     import to_percent, trim_df
from .stimutils import (exp, db, stim, present_trial, present_break,
    show_resp_rules, textscreen, present_feedback, present_training,
    give_training_db, Instructions, onflip_work, clear_port, break_checker)
from .viz import plot_quest_plus

if exp['use trigger']:
    from ctypes import windll


# make mouse invisible
stim['window'].mouseVisible = False

# set logging
dm = DataManager(exp)
exp = dm.update_exp(exp)
exp['numTrials'] = 560 # ugly hack, CHANGE
log_path = dm.give_path('l', file_ending='log')
lg = logging.LogFile(f=log_path, level=logging.WARNING, filemode='w')

monkey = None
if exp['debug']:
    resp_mapping = exp['keymap']
    monkey = PsychometricMonkey(
        response_mapping=resp_mapping, intensity_var='opacity',
        stimulus_var='orientation')


# create object for updating experimenter about progress
exp_info = ExperimenterInfo(exp, stim, main_text_pos=(0, 0.90),
                            sub_text_pos=(0, 0.78))


# TODO: add eeg baseline (resting-state)!
# TODO: add some more logging?


# INSTRUCTIONS
# ------------
if exp['run instruct']:
    instr = Instructions('instructions.yaml', auto=exp['debug'])
    instr.present(stop=8)
    show_resp_rules(exp=exp, text=u"Tak wygląda ekran przerwy.",
                    auto=exp['debug'])
    instr.present(stop=12)
    # "are there any questions":
    qst_gui = AnyQuestionsGUI(exp, stim, auto=exp['debug'])
    qst_gui.run()
    instr.present(stop=13)


# show response rules:
msg = (u"Zaraz rozpocznie się trening.\nPrzygotuj się.\nPamiętaj o "
       u"pozycji palców na klawiaturze.")
show_resp_rules(exp=exp, text=msg, auto=exp['debug'])

# TRAINING
# --------
if exp['run training'] and not exp['debug']:

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
    addtxt = (u'Szybkość prezentacji bodźców zostaje zwiększona.'
              u'\nAby przejść dalej naciśnij spację.')

    for s, c in zip(exp['train slow'], exp['train corr']):
        # present current training block until correctness is achieved
        df, current_corr = present_training(exp=slow, slowdown=s, corr=c,
                                            monkey=monkey, auto=exp['debug'])
        current_block += 1

        # update experimenter info:
        exp_info.training_info([current_block, num_training_blocks],
                               current_corr)

        # show info for the subject:
        if s == 1:
            addtxt = (u'Koniec treningu.\nAby przejść dalej '
                      u'naciśnij spację.')
        now_txt = txt + addtxt
        textscreen(now_txt.format(to_percent(current_corr)), auto=exp['debug'])
        show_resp_rules(exp=exp, auto=exp['debug'])

        # concatenate training df's
        df_train.append(trim_df(df))

    # save training database:
    df_train = pd.concat(df_train)
    df_train.reset_index(drop=True, inplace=True)
    df_train.to_excel(dm.give_path('a'))


# Contrast fitting
# ----------------
# staircase, QuestPlus, then QPThreshold
omit_first_fitting_steps = exp['start at thresh fitting'] and exp['debug']
if exp['run fitting'] and not omit_first_fitting_steps:
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
    current_trial = 1
    max_trials = exp['staircase trials']
    staircase = StairHandler(0.8, nTrials=max_trials, nUp=1, nDown=2,
                             nReversals=6, minVal=0.001, maxVal=2,
                             stepSizes=[0.1, 0.1, 0.05, 0.05, 0.025, 0.025],
                             stepType='lin')

    for contrast in staircase:
        exp_info.blok_info(u'procedura schodkowa', [current_trial, max_trials])

        # setup stimulus and present trial
        exp['opacity'] = [contrast, contrast]
        core.wait(0.5) # fixed pre-fix interval
        present_trial(current_trial, db=fitting_db, exp=exp, monkey=monkey)
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
            show_resp_rules(exp=exp, auto=exp['debug'])
            stim['window'].flip()


if exp['run fitting'] and not omit_first_fitting_steps:
    # QUEST+
    # ------
    # next, after about 25 trials we start main fitting procedure - QUEST+

    # init quest plus
    start_contrast = staircase._nextIntensity
    stim_params = from_db(np.arange(-20, 3.1, 0.35)) # -20 dB is about 0.01

    model_params = [exp['thresholds'], exp['slopes'], exp['lapses']]
    qp = QuestPlus(stim_params, model_params, function=weibull)
    min_idx = np.abs(stim_params - start_contrast).argmin()
    contrast = stim_params[min_idx]

    # args for break-related stuff
    qp_refresh_rate = sample([3, 4, 5], 1)[0]
    img_name = op.join(exp['data'], 'quest_plus_panel.png')
    plot_fun = lambda x: plot_quest_plus(x)
    df_save_path = dm.give_path('b')

    # update experimenters view:
    block_name = u'Quest Plus, część I'
    exp_info.blok_info(block_name, [0, 100])

    # remind about the button press mappings
    show_resp_rules(exp=exp, auto=exp['debug'])
    stim['window'].flip()

    for trial in range(exp['QUEST plus trials']):
        # CHECK if blok_info flips the screen, better if not...
        exp_info.blok_info(block_name, [trial + 1, 100])

        # setup stimulus and present trial
        exp['opacity'] = [contrast, contrast]
        core.wait(0.5) # fixed pre-fix interval
        present_trial(current_trial, db=fitting_db, exp=exp, monkey=monkey)
        stim['window'].flip()

        # set trial type, get response and inform staircase about it
        fitting_db.loc[current_trial, 'trial_type'] = 'Quest+'
        response = fitting_db.loc[current_trial, 'ifcorrect']
        qp.update(contrast, response)
        contrast = qp.next_contrast()

        # check for and perform break-related stuff
        qp_refresh_rate = break_checker(
            stim['window'], exp, fitting_db, exp_info, lg, current_trial,
            qp_refresh_rate=qp_refresh_rate, plot_fun=plot_fun, plot_arg=qp,
            dpi=120, img_name=img_name, df_save_path=df_save_path)
        current_trial += 1

    # saving quest may seem unnecessary - posterior can be reproduced
    # from trials, nevertheless it is useful for debugging
    posterior_filename = dm.give_path('posterior', file_ending='npy')
    np.save(posterior_filename, qp.posterior)

if exp['run fitting']:
    # threshold fitting
    # -----------------

    # if omit_first_fitting_steps:
    #         read density and trials from disc
    #        (or simulate without screen update)

    # initialize further threshold optimization
    trimmed_df = trim_df(fitting_db)
    corrs, qps = init_thresh_optim(trimmed_df, qp, model_params)
    block_name = u'QuestPlus, część II'
    fig, ax = plt.subplots()

    plot_fun = lambda x: plot_threshold_entropy(x, corrs=corrs, axis=ax).figure
    img_name = op.join(exp['data'], 'quest_plus_thresholds.png')
    qp_refresh_rate = break_checker(
        stim['window'], exp, fitting_db, exp_info, lg, 1,
        qp_refresh_rate=1, plot_fun=plot_fun, plot_arg=qps,
        dpi=120, img_name=img_name, df_save_path=df_save_path)
    ax.clear()

    # optimize thresh...
    for trial in range(exp['thresh opt trials']):
        # select contrast
        posteriors = [qp.get_posterior().sum(axis=(1, 2)) for qp in qps]
        posterior_peak = [posterior.max() for posterior in posteriors]
        optimize_threshold = np.array(posterior_peak).argmin()
        contrast = qps[optimize_threshold].next_contrast(axis=0)

        # CHECK if blok_info flips the screen, better if not...
        exp_info.blok_info(block_name, [trial + 1, exp['thresh opt trials']])

        # setup stimulus and present trial
        exp['opacity'] = [contrast, contrast]
        core.wait(0.5) # fixed pre-fix interval
        present_trial(current_trial, db=fitting_db, exp=exp, monkey=monkey)
        stim['window'].flip()

        # set trial type, get response and inform staircase about it
        fitting_db.loc[current_trial, 'trial_type'] = 'Quest+ thresholds'
        response = fitting_db.loc[current_trial, 'ifcorrect']
        for qp in qps:
            qp.update(contrast, response)
        current_trial += 1

        # check for and perform break-related stuff
        qp_refresh_rate = break_checker(
            stim['window'], exp, fitting_db, exp_info, lg, current_trial,
            qp_refresh_rate=qp_refresh_rate, plot_fun=plot_fun, plot_arg=qps,
            dpi=120, img_name=img_name, df_save_path=df_save_path)
        ax.clear()

    # save fitting dataframe
    fitting_db = trim_df(fitting_db)
    fitting_db.to_excel(dm.give_path('b'))

    # saving quest may seem unnecessary - posterior can be reproduced
    # from trials, nevertheless we save the posterior as numpy array
    posterior_filename = dm.give_path('posterior_thresh', file_ending='npy')
    np.save(posterior_filename, qps[2].posterior)


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
        # params = qp.get_fit_params()
        contrasts.append(params[0])

    lg.write('final contrast steps: {}'.format(contrasts))

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
        present_trial(i, exp=exp, db=db_c, use_exp=False, monkey=monkey)
        stim['window'].flip()

        # present break
        if i % exp['break after'] == 0:
            # save data before every break
            temp_db = trim_df(db_c.copy())
            temp_db.to_excel(dm.give_path('c'))
            # break and refresh keyboard mapping
            present_break(i, exp=exp, auto=exp['debug'])
            show_resp_rules(exp=exp, auto=exp['debug'])
            stim['window'].flip()

        # update experimenter
        exp_info.blok_info(u'główne badanie', [i, exp['numTrials']])
        # - [ ] add some figure with correctness for different contrast steps

    db_c.to_excel(dm.give_path('c'))

# goodbye!
# - [ ] some thanks etc. here!
core.quit()
