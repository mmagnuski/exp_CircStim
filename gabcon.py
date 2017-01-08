# -*- coding: utf-8 -*-

# add description


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
if psychopy_version.startswith('1.84'):
    from psychopy.visual import shaders
else:
    from psychopy import _shadersPyglet as shaders

shaders.fragFBOtoFrame = fragFBOtoFramePatched

# other imports
# -------------
from psychopy  import visual, core, event, logging
from psychopy.data import QuestHandler

import os
from random import sample

import numpy  as np
import pandas as pd

from exputils  import (plot_Feedback, create_database,
    ContrastInterface, DataManager, ExperimenterInfo,
    AnyQuestionsGUI, ms2frames, getFrameRate)
from weibull   import fitw, get_new_contrast, correct_weibull
from utils     import (to_percent, round2step, trim_df, grow_sample,
                       time_shuffle)
from stimutils import (exp, db, stim, present_trial,
    present_break, show_resp_rules, textscreen,
    present_feedback, present_training, trim,
    give_training_db, Instructions, Stepwise,
    onflip_work, clear_port)

if os.name == 'nt' and exp['use trigger']:
    from ctypes import windll

# set logging
dm = DataManager(exp)
exp = dm.update_exp(exp)
exp['numTrials'] = 500 # ugly hack, change
log_path = dm.give_path('l', file_ending='log')
lg = logging.LogFile(f=log_path, level=logging.WARNING, filemode='w')


# TODO: add eeg baseline (resting-state)!
# if fitting completed -> use data
# if c part done -> use data
# check via dm.give_previous_path('b') etc.

# TODO: turn prints to logging
# TODO: add some more logging?

# create object for updating experimenter about progress
exp_info = ExperimenterInfo(exp, stim)


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


# Contrast fitting - QUEST staircase
# ----------------------------------
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

    # init stepwise contrast adjustment
    step = exp['step until']
    exp['opacity'] = [1., 1.]
    fitting_db = give_training_db(db, slowdown=1)

    # update experimenters view:
    block_name = 'schodkowe dopasowywanie kontrastu'
    exp_info.blok_info(block_name, [0, step[0]])


    # remind about the button press mappings
    show_resp_rules(exp=exp)
    stim['window'].flip()

    current_trial = 1
    continue_fitting = True
    kwargs = dict(gamma=0.01, nTrials=60, minVal=0., maxVal=2.)
    staircases = [QuestHandler(0.7, 0.1, pThreshold=p, **kwargs)
                  for p in [0.55, 0.95]]
    active_staircases = [0, 1]

    # TODOs:
    # - [ ] add break screen (response mappings)
    # - [x] check StopIteration for both staircases and adapt
    # - [x] stop fitting once both staircases finish
    # - [ ] add saveAsPickle to investigate fitted objs
    while continue_fitting:
        # choose staircase
        chosen_ind = sample(active_staircases, 1)[0]
        current_staircase = staircases[chosen_ind]
        try:
            contrast = current_staircase.next()
        except StopIteration:  # we got a StopIteration error
            active_staircases.pop(chosen_ind)
            continue_fitting = len(active_staircases) > 0
            continue

        # setup stimulus and present trial
        exp['opacity'] = [contrast, contrast]
        core.wait(0.5) # fixed pre-fix interval
        present_trial(current_trial, db=fitting_db, exp=exp)
        stim['window'].flip()

        # get response and inform QUEST about it
        response = fitting_db.loc[current_trial, 'ifcorrect']
        current_staircase.addResponse(response)
        current_trial += 1

    # can now access 1 of 3 suggested threshold levels
    # staircase.mean()

    # save fitting dataframe
    fitting_db = trim_df(fitting_db)
    fitting_db.to_excel(dm.give_path('b'))


# EXPERIMENT - part c
# -------------------
if exp['run main c']:
    # instructions
    if exp['run instruct']:
        instr.present(stop=16)

    # get contrast from fitting
    if 'fitting_db' not in locals():
        prev_pth = dm.give_previous_path('b')
        print('fitting_db not found, loading {}...'.format(prev_pth))
        fitting_db = pd.read_excel(prev_pth)
        print(fitting_db.head(10))
        # fitting_db = pd.read_excel(
        #     os.path.join('data', 'testing_miko_01_b_1.xls'))

    # setup stuff for GUI:
    if not 'window2' in stim:
        stim['window'].blendMode = 'avg'
    # check with drawing target...

    # make sure weibull exists
    set_im = False
    if 'w' not in locals():
        num_trials = fitting_db.shape[0]
        ind = np.r_[25:num_trials]
        w = fitw(fitting_db, ind, init_params=[1., 1., 1.])
    stim = plot_Feedback(stim, w, exp['data'])

    interf = ContrastInterface(stim=stim, exp=exp, df=fitting_db, weibull=w,
                               set_image_size=set_im, contrast_method='10steps')
    continue_fitting = interf.loop()

    if not 'window2' in stim:
        stim['window'].blendMode = 'add'

    # check the fit and weibull params
    print('final weibull: ', interf.weibull)
    print('final params: ', interf.params)
    print('final num_trials: ', interf.num_trials)

    # num_trials = fitting_db.shape[0]
    # w = fitw(fitting_db, range(num_trials-100, num_trials))
    contrast_range = interf.weibull.get_threshold(exp['corrLims'])
    contrast_steps = np.linspace(contrast_range[0],
        contrast_range[1], exp['opac steps'])

    print('contrast range: ', contrast_range)
    print('contrast steps: ', contrast_steps)

    db_c = create_database(exp, combine_with=('opacity',
        contrast_steps), rep=13)
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
        if (i) % exp['break after'] == 0:
            # save data before every break
            db_c.to_excel(dm.give_path('c'))
            # break and refresh keyboard mapping
            present_break(i, exp=exp)
            show_resp_rules(exp=exp)
            stim['window'].flip()

        # update experimenter
        exp_info.blok_info(u'główne badanie, część I', [i, exp['numTrials']])

    db_c.to_excel(dm.give_path('c'))


# EXPERIMENT - part t
# -------------------
if exp['run main t']:

    exp['break after'] == 7 # because trials are longer now

    # get contrast from contrast part
    if 'db_c' not in locals():
        print('contrast database not found, loading from disk...')
        db_c = pd.read_excel(dm.give_previous_path('c'))
        print(db_c.head(10))


    # ---CHANGE HERE---

    # FinalFitGUI - to set about 75% thresh
    # setup
    # setup stuff for GUI:
    if not 'window2' in stim:
        stim['window'].blendMode = 'avg'
    # check with drawing target...

    # make sure weibull exists
    set_im = False
    if 'w' not in locals():
        num_trials = db_c.shape[0]
        ind = np.r_[25:num_trials]
        w = fitw(db_c, ind, init_params=[1., 1., 1.])
    stim = plot_Feedback(stim, w, exp['data'])

    interf = ContrastInterface(stim=stim, exp=exp, df=db_c, weibull=w,
                               set_image_size=set_im, corr_steps=[0.7],
                               contrast_method=None, num_trials=db_c.shape[0])
    continue_fitting = interf.loop()

    if not 'window2' in stim:
        stim['window'].blendMode = 'add'

    # get contrast and prepare trials dataframe
    opacity = interf.weibull.get_threshold([0.7])[0]
    times = time_shuffle(start=1., end=5., every=0.2, times=4)
    times = ms2frames(times * 1000, exp['frm']['time'])
    db_t = create_database(exp, rep=1, combine_with=('fixTime', times))
    db_t.loc[:, 'opacity'] = opacity

    exp['numTrials'] = len(db_t.index) # TODO/CHECK - why is this needed?

    if exp['run instruct']:
        instr.present()

    # signal onset of 'time' part
    if exp['use trigger']:
        core.wait(0.05)
        onflip_work(exp['port'], 'time')
        core.wait(0.1)
        clear_port(exp['port'])

    # main loop
    for i in range(1, db_t.shape[0] + 1):
        core.wait(0.5) # pre-fixation time is always the same
        present_trial(i, exp=exp, db=db_t, use_exp=False)
        stim['window'].flip()

        # present break
        if (i) % exp['break after'] == 0:
            # save data before every break
            db_t.to_excel(dm.give_path('t'))
            # break and refresh keyboard mapping
            present_break(i, exp=exp)
            show_resp_rules(exp=exp)
            stim['window'].flip()

        # update experimenter
        exp_info.blok_info(u'główne badanie, część II', [i, exp['numTrials']])

    # save data before quit
    db_t.to_excel(dm.give_path('t'))

# goodbye!
core.quit()
