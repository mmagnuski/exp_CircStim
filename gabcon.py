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

from .baseline import run as run_baseline
from .exputils  import (plot_Feedback, create_database, DataManager,
                        ExperimenterInfo, AnyQuestionsGUI)
from .weibull   import (Weibull, QuestPlus, weibull_db, PsychometricMonkey,
                        init_thresh_optim, from_db, to_db, weibull)
from .utils     import to_percent, trim_df
from .stimutils import (exp, db, stim, present_trial, present_break,
    show_resp_rules, textscreen, present_feedback, present_training,
    give_training_db, Instructions, onflip_work, clear_port, break_checker,
    forced_break, final_info)
from .viz import plot_quest_plus, plot_threshold_entropy

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
subj_id = exp['participant']['ID']

monkey = None
if exp['debug']:
    resp_mapping = exp['keymap']
    monkey = PsychometricMonkey(
        response_mapping=resp_mapping, intensity_var='opacity',
        stimulus_var='orientation')


# create object for updating experimenter about progress
exp_info = ExperimenterInfo(exp, stim, main_text_pos=(0, 0.90),
                            sub_text_pos=(0, 0.78))

def general_trigger(port, code):
    if exp['use trigger']:
        core.wait(0.05)
        onflip_work(port, code=code)
        stim['window'].flip()
        core.wait(0.05)
        clear_port(port)


# eeg baseline (resting-state)
if exp['run baseline1']:
    run_baseline(stim['window'], exp, segment_time=70., debug=exp['debug'],
                 exp_info=exp_info)
    exp_info.general_info(u'skończył się baseline\nmożna odłączyć głośniki')


# INSTRUCTIONS
# ------------
if exp['run instruct']:
    instr = Instructions(r'instr\instructions.yaml', auto=exp['debug'],
                         exp_info=exp_info)
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
if exp['run training']:

    # signal onset of training
    general_trigger(exp['port'], 'training')
    train_pth = dm.give_path('a')

    # set things up
    slow = exp.copy()
    df_train = list()
    num_training_blocks = len(exp['train slow'])

    txt = u'Twoja poprawność: {}\nOsiągnięto wymaganą poprawność.\n'
    addtxt = (u'Szybkość prezentacji bodźców zostaje zwiększona.'
              u'\nAby przejść dalej naciśnij spację.')

    trial = 1
    contrast = 1.
    current_block = 0
    for slowdown, corr in zip(exp['train slow'], exp['train corr']):
        # present current training block until correctness is achieved
        train_db = give_training_db(db, slowdown=slowdown)
        df, current_corr, contrast = present_training(
            trial, train_db, exp=slow, slowdown=slowdown, corr=corr,
            monkey=monkey, exp_info=exp_info, contrast=contrast,
            block_num=[current_block, num_training_blocks])
        current_block += 1

        # update experimenter info:
        exp_info.training_info([current_block, num_training_blocks],
                               current_corr)

        # show info for the subject:
        if slowdown == 1:
            addtxt = (u'Koniec treningu.\nAby przejść dalej '
                      u'naciśnij spację.')
        now_txt = txt + addtxt
        textscreen(now_txt.format(to_percent(current_corr)), auto=exp['debug'])
        show_resp_rules(exp=exp, auto=exp['debug'])

        if contrast > 1.:
            contrast -= 0.5

        # concatenate training df's
        df_train.append(trim_df(df))

        # save training database:
        df_train_save = pd.concat(df_train)
        df_train_save.reset_index(drop=True, inplace=True)
        df_train_save.to_excel(train_pth)


# Contrast fitting
# ----------------
# staircase, QuestPlus, then QPThreshold
omit_first_fitting_steps = exp['start at thresh fitting'] and exp['debug']
if exp['run fitting'] and not omit_first_fitting_steps:
    # some instructions
    if exp['run instruct']:
        instr.present(stop=15)

    # send start trigger:
    general_trigger(exp['port'], 'fitting')

    # init fitting db
    fitting_db = give_training_db(db, slowdown=1)

    # Staircase
    # ---------
    # we start with staircase to make sure subjects familiarize themselves
    # with adapting contrast regime before the main fitting starts
    current_trial = 1
    max_trials = exp['staircase trials']
    staircase = StairHandler(0.8, nTrials=max_trials, nUp=1, nDown=2,
                             nReversals=6, minVal=0.01, maxVal=2.,
                             stepSizes=[0.1, 0.1, 0.05, 0.05, 0.025, 0.025],
                             stepType='lin')

    for contrast in staircase:
        # never go longer than 35 trials
        if current_trial > 35:
           break

        exp_info.blok_info(u'procedura schodkowa', [current_trial, max_trials])
        core.wait(0.5) # fixed pre-fix interval
        present_trial(current_trial, db=fitting_db, contrast=contrast, exp=exp,
                      monkey=monkey)
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

    # init quest plus
    stim_params = from_db(np.arange(-20, 3.1, 0.35)) # -20 dB is about 0.01
    model_params = [exp['thresholds'], exp['slopes'], exp['lapses']]
    qp = QuestPlus(stim_params, model_params, function=weibull)

    # find starting contrast
    start_contrast = staircase._nextIntensity
    min_idx = np.abs(stim_params - start_contrast).argmin()
    contrast = stim_params[min_idx]

    # set up priors
    logfun = lambda x, th, slp: 0.8 / (1 + np.exp(-slp * (x - th))) + 0.2
    x = np.linspace(0, 1, num=len(model_params[1]))
    y1 = logfun(x, 0.1, 20)
    y2 = logfun(x, 0.9, -20)
    slope_prior = y1 * y2
    lapse_prior = np.array([1., 1., 1., 1., 0.8, 0.5])
    threshold_prior = np.ones(len(model_params[0]))

    p1, p2, p3 = np.meshgrid(slope_prior, threshold_prior, lapse_prior)
    priors = p1 * p2 * p3
    priors /= priors.sum()
    qp.posterior = priors.ravel()

    # args for break-related stuff
    qp_refresh_rate = sample([3, 4, 5], 1)[0]
    img_name = op.join(exp['data'], '{}_quest_plus_panel.png'.format(subj_id))
    plot_fun = lambda x: plot_quest_plus(x)
    df_save_path = dm.give_path('b')

    # update experimenters view:
    block_name = u'Quest Plus, część I'
    exp_info.blok_info(block_name, [0, exp['QUEST plus trials']])

    # remind about the button press mappings
    show_resp_rules(exp=exp, auto=exp['debug'])
    stim['window'].flip()

    for trial in range(exp['QUEST plus trials']):
        # CHECK if blok_info flips the screen, better if not...
        exp_info.blok_info(block_name, [trial + 1, exp['QUEST plus trials']])
        core.wait(0.5) # fixed pre-fix interval
        present_trial(current_trial, db=fitting_db, contrast=contrast, exp=exp,
                      monkey=monkey)
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

    # forced break
    forced_break(auto=exp['debug'], exp_info=exp_info)
    show_resp_rules(exp=exp, auto=exp['debug'])
    present_break(current_trial, exp=exp, auto=exp['debug'])

    # THRESHOLD FITTING
    # -----------------
    block_name = u'Quest Plus, część II'
    corrs = np.linspace(0.6, 0.9, num=5)
    def get_contrasts(qp, corrs):
        weib = Weibull(kind='weibull')
        weib.params = qp.get_fit_params()
        return weib.get_threshold(corrs)

    contrasts = get_contrasts(qp, corrs)
    lg.write('Contrast steps after {} trials: {}\n'.format(trial, contrasts))
    use_contrasts = np.tile(np.asarray(contrasts), 2)
    np.random.shuffle(use_contrasts)

    trial = 0
    max_trials = exp['thresh opt trials']
    while trial + 1 <= max_trials:
        for contrast in use_contrasts:
            # CHECK if blok_info flips the screen, better if not...
            exp_info.blok_info(block_name, [trial + 1, max_trials])

            # setup stimulus and present trial
            core.wait(0.5) # fixed pre-fix interval
            present_trial(current_trial, db=fitting_db, contrast=contrast,
                          exp=exp, monkey=monkey)
            stim['window'].flip()

            # set trial type, get response and inform staircase about it
            fitting_db.loc[current_trial, 'trial_type'] = 'Quest+'
            response = fitting_db.loc[current_trial, 'ifcorrect']
            qp.update(contrast, response, approximate=True)

            # check for and perform break-related stuff
            qp_refresh_rate = break_checker(
                stim['window'], exp, fitting_db, exp_info, lg, current_trial,
                qp_refresh_rate=qp_refresh_rate, plot_fun=plot_fun, plot_arg=qp,
                dpi=120, img_name=img_name, df_save_path=df_save_path)
            current_trial += 1
            trial += 1

        # after each block of 10 trials (2 * 5 steps) - reevaluate QP
        contrasts = get_contrasts(qp, corrs)
        lg.write('Contrast steps after {} trials: {}\n'.format(
            trial, contrasts))
        use_contrasts = np.tile(np.asarray(contrasts), 2)
        np.random.shuffle(use_contrasts)


# EXPERIMENT - part c
# -------------------
if exp['run main c']:
    # instructions
    if exp['run instruct']:
        instr.present(stop=16)

    # get contrast thresholds from quest plus:
    corrs = np.linspace(0.6, 0.9, num=5)
    contrasts = get_contrasts(qp, corrs)
    lg.write('contrast steps at the beginning of procedure: {}\n'.format(contrasts))

    # set up break plots
    qp_refresh_rate = sample([3, 4, 5], 1)[0]
    img_name = op.join(exp['data'], '{}_final_proc_panel.png'.format(subj_id))

    def wb_plot(wb, lst):
        df = lst[0]
        df = trim_df(df.copy())
        wb.x = df.opacity.values.copy()
        wb.y = df.ifcorrect.values.copy()
        wb.orig_y = df.ifcorrect.values.copy()
        return wb.plot(mean_points=True, contrast_steps=lst[1]).figure

    weib = Weibull(kind='weibull')
    weib.params = qp.get_fit_params()
    plot_fun = lambda x: wb_plot(weib, x)
    df_save_path = dm.give_path('c')

    # 32 repetitions * 4 angles * 5 steps = 640 trials
    db_c = create_database(exp, combine_with=('opacity', contrasts), rep=32,
                           shuffle_in_reps=True, numerate_steps=True)
    exp['numTrials'] = len(db_c.index)
    contrasts = np.asarray(contrasts)

    # signal that main proc is about to begin
    general_trigger(exp['port'], 'contrast')

    # main loop
    for i in range(1, db_c.shape[0] + 1):
        core.wait(0.5) # pre-fixation time is always the same
        present_trial(i, exp=exp, db=db_c, monkey=monkey, contrast=None)
        stim['window'].flip()

        # we still update QP
        contrast = db_c.loc[i, 'opacity']
        response = db_c.loc[i, 'ifcorrect']
        qp.update(contrast, response, approximate=True)

        # break handling
        qp_refresh_rate = break_checker(
            stim['window'], exp, db_c, exp_info, lg, i,
            qp_refresh_rate=1000, plot_fun=plot_fun,
            plot_arg=[db_c, contrasts], dpi=120, img_name=img_name,
            df_save_path=df_save_path, show_completed=True,
            show_correctness=True, use_forced_break=True)

        # update experimenter
        exp_info.blok_info(u'główne badanie', [i, exp['numTrials']])

        # if qp gives very different steps - change
        if (i % 20 == 0) and (i <= 120):
            prev_diffs = np.abs(np.diff(contrasts))
            prev_diffs = np.append(prev_diffs, prev_diffs[-1])
            new_contrasts = np.asarray(get_contrasts(qp, corrs))
            contrasts_differences = np.abs(new_contrasts - contrasts)
            change = contrasts_differences > (prev_diffs * 0.33)

            if change.any():
                msg = 'Changed final contrast steps after trial {} to: {}\n'
                lg.write(msg.format(i, contrasts))
                contrasts[change] = new_contrasts[change]
                step = db_c.loc[i + 1:, 'step'].values
                opacity = db_c.loc[i + 1:, 'opacity'].values
                for idx in range(len(contrasts)):
                    if change[idx]:
                        msk = step == idx + 1
                        opacity[msk] = contrasts[idx]
                db_c.loc[i + 1:, 'opacity'] = opacity

    db_c.to_excel(dm.give_path('c'))

# goodbye!
higher_steps = db_c.query('step > 2')
corr = higher_steps.ifcorrect.mean()
payout = int(round(75. * corr))
final_info(corr, payout, auto=exp['debug'], exp_info=exp_info)
core.quit()
