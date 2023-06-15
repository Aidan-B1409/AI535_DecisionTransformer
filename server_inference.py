import gymnasium as gym
import torch
import numpy as np
import argparse
from transformers import DecisionTransformerModel
import matplotlib.pyplot as plt
import os
import shutil


def get_action(model, state_dim, act_dim, states, actions, rewards, returns_to_go, timesteps):
    # This implementation does not condition on past rewards
    
    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)
    
    # The prediction is conditioned on up to 20 previous time-steps
    states = states[:, -model.config.max_length :]
    actions = actions[:, -model.config.max_length :]
    returns_to_go = returns_to_go[:, -model.config.max_length :]
    timesteps = timesteps[:, -model.config.max_length :]
    
    # pad all tokens to sequence length, this is required if we process batches
    padding = model.config.max_length - states.shape[1]
    attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
    states = torch.cat([torch.zeros((1, padding, state_dim)), states], dim=1).float()
    actions = torch.cat([torch.zeros((1, padding, act_dim)), actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)
    
    # perform the prediction
    state_preds, action_preds, return_preds = model(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,)
    return action_preds[0, -1]





PICKANDPLACE = 'pick'
SLIDE = 'slide'
PUSH = 'push'
REACH = 'reach'

ALLEXPERT = 'expert'
ALLRANDOM = 'random'
FIFTYRANDOMFIFTYEXPERT = '5050'
NINETYRANDOMTENEXPERT = '9010'


def inference(modelpath, task, data, num_runs, MAX_EPISODE_LENGTH):

    if task == PICKANDPLACE:
        ENV_NAME = 'FetchPickAndPlaceDense-v2'
        if data == ALLEXPERT:
            TARGET_RETURN = -7.915132197510427
            state_mean = np.array([ 1.34785880e+00,  7.48924685e-01,  4.69636954e-01,  1.34566766e+00,
            7.47857853e-01,  4.37020973e-01, -2.19113409e-03, -1.06683152e-03,
            -3.26159799e-02,  3.81253912e-02,  3.80331808e-02, -1.59406920e-02,
            -3.86805341e-03, -3.23361478e-03, -4.33292879e-05, -1.00737217e-04,
            2.66919555e-04, -5.11537041e-04,  2.12608221e-04, -2.43298010e-04,
            1.31203537e-04, -3.23062378e-05, -6.47135383e-04, -5.04662921e-05,
            -1.64692124e-05,  1.34566766e+00,  7.47857853e-01,  4.37020973e-01,
            1.34179681e+00,  7.49364571e-01,  5.36314552e-01])

            state_std = np.array([0.08884744, 0.08711861, 0.05129468, 0.10183314, 0.09674642, 0.05762528,
            0.05290669, 0.04197282, 0.056866,   0.01474597, 0.01476374, 0.74228999,
            0.28402358, 0.62257303, 0.01128793, 0.01191195, 0.0130018,  0.11470533,
            0.05730912, 0.09737996, 0.01267085, 0.01232798, 0.01309373, 0.01851943,
            0.01860757, 0.10183314, 0.09674642, 0.05762528, 0.08656078, 0.08653932,
            0.14519501])

        if data == ALLRANDOM:
            TARGET_RETURN = -11.763010237967496
            state_mean = np.array([ 1.33442257e+00,  7.49169987e-01,  5.39591570e-01,  1.34119316e+00,
            7.49051326e-01,  4.23625392e-01,  6.77059431e-03, -1.18660276e-04,
            -1.15966174e-01,  3.54246684e-02,  3.54193653e-02, -4.86994685e-04,
            7.38104336e-04, -6.94089334e-04,  2.19083496e-04, -1.01351502e-05,
            -3.42397615e-04,  3.82406002e-05,  1.00514564e-04,  1.63552946e-06,
            -2.12184692e-04,  8.97643079e-06,  2.68600248e-04,  6.56471075e-04,
            6.56602442e-04,  1.34119316e+00,  7.49051326e-01,  4.23625392e-01,
            1.34168463e+00,  7.49133061e-01,  5.37543541e-01])

            state_std = np.array( [0.08217741, 0.08999253, 0.07556938, 0.10395619, 0.10663859, 0.02130656,
            0.13153449, 0.13813253, 0.07754545, 0.01816918, 0.01816959, 0.24202244,
            0.11174591, 0.19257261, 0.01458929, 0.01507367, 0.01500477, 0.02984707,
            0.02323423, 0.02030909, 0.014562, 0.01492165, 0.01480703, 0.02423637,
            0.02424095, 0.10395619, 0.10663859, 0.02130656, 0.08644656, 0.08649478,
            0.14531839])

        if data == NINETYRANDOMTENEXPERT:
            TARGET_RETURN = -11.351776023352835
            state_mean = np.array([ 1.33576719e+00,  7.49005425e-01,  5.32943181e-01,  1.34193661e+00,
            7.48860252e-01,  4.25129895e-01,  6.16941379e-03, -1.45172945e-04,
            -1.07813282e-01,  3.56745821e-02,  3.56645483e-02, -3.07219780e-03,
            1.23838434e-03, -3.01875436e-04,  2.02715851e-04, -2.05506498e-05,
            -2.89121396e-04, -4.51548915e-05,  1.34977044e-04,  3.33881830e-05,
            -1.76764372e-04,  1.58921372e-06,  1.78412368e-04,  5.84377591e-04,
            5.87263809e-04,  1.34193661e+00,  7.48860252e-01,  4.25129895e-01,
            1.34159644e+00,  7.49297063e-01,  5.37375327e-01])
            state_std = np.array([0.08280165, 0.08947497, 0.07614444, 0.10368418, 0.1055017,  0.0281082,
            0.12584239, 0.13161694, 0.07943101, 0.01786873, 0.01786965, 0.32862691,
            0.13885715, 0.26356059, 0.01428553, 0.01477602, 0.01483672, 0.04780163,
            0.02869608, 0.03551986, 0.01437986, 0.01466664, 0.01466081, 0.0237392,
            0.0237491,  0.10368418, 0.1055017,  0.0281082,  0.0864777,  0.08659862,
            0.14527066])




    if task == SLIDE:
        ENV_NAME = 'FetchSlideDense-v2'
        if data == ALLEXPERT:
            TARGET_RETURN = -20.37208572901309
            state_mean = np.array([ 1.03198449e+00,  7.67391669e-01,  4.48380007e-01,  1.16446727e+00,
            7.39030704e-01,  3.74656022e-01,  1.32482780e-01, -2.83609647e-02,
            -7.37239848e-02, -4.13081264e-08,  2.98508486e-05, -2.05454365e-03,
            4.37844944e-03, -2.39356986e-01,  5.50067253e-03, -1.36199968e-03,
            -3.18877202e-03,  5.24988147e-05, -4.84119974e-05, -5.45745558e-03,
            8.35787183e-04,  5.24802195e-04,  9.66417456e-04,  9.66506515e-04,
            1.14018054e-03,  1.16446727e+00,  7.39030704e-01,  3.74656022e-01,
            1.39537431e+00,  7.48741872e-01,  4.14018929e-01])
            state_std = np.array([1.02980290e-01, 1.52279769e-01, 3.84293342e-02, 2.47214860e-01,
            2.64022039e-01, 1.15842538e-01, 1.95383452e-01, 1.70012704e-01,
            1.21182642e-01, 1.28619109e-06, 2.07812745e-04, 3.58925630e-01,
            1.93586082e-01, 6.31476887e-01, 1.37233949e-02, 1.36091919e-02,
            1.57747355e-02, 7.13571032e-02, 7.11392451e-02, 5.46714361e-02,
            1.15381330e-02, 1.17473520e-02, 9.31058317e-03, 2.10811269e-03,
            2.26187488e-03, 2.47214860e-01, 2.64022039e-01, 1.15842538e-01,
            1.73126759e-01, 1.73293777e-01, 1.00000000e-06])

        if data == ALLRANDOM:
            TARGET_RETURN = -22.685298620755212
            state_mean = np.array( [ 9.87627328e-01,  7.49047457e-01,  4.76664494e-01,  9.96046980e-01,
            7.49869819e-01,  4.06172363e-01,  8.41965159e-03,  8.22361800e-04,
            -7.04921316e-02, -4.13081264e-08,  2.98508486e-05, -1.41721757e-04,
            5.13441923e-03, -1.70186609e-01,  3.81669466e-04,  1.75487647e-05,
            -2.05528203e-03,  3.26702049e-04,  3.66943478e-04, -5.64767177e-03,
            -3.65780476e-04,  1.24204526e-05,  1.65120556e-03,  1.16398136e-03,
            1.16711809e-03,  9.96046980e-01,  7.49869819e-01,  4.06172363e-01,
            1.39617884e+00,  7.48535122e-01,  4.14018929e-01])

            state_std = np.array( [8.16953070e-02, 8.83932667e-02, 5.93734312e-02, 1.21142988e-01,
            1.21423287e-01, 5.37759647e-02, 1.40845671e-01, 1.44711994e-01,
            8.09763923e-02, 1.28619109e-06, 2.07812745e-04, 2.41040468e-01,
            1.34358787e-01, 5.03372668e-01, 1.46951371e-02, 1.50174489e-02,
            1.49130271e-02, 4.69711842e-02, 4.75029181e-02, 2.70186715e-02,
            1.40813481e-02, 1.44638875e-02, 1.36968285e-02, 2.27573406e-03,
            2.27987990e-03, 1.21142988e-01, 1.21423287e-01, 5.37759647e-02,
            1.72722875e-01, 1.73165900e-01, 1.00000000e-06])
       
        if data == NINETYRANDOMTENEXPERT:
            TARGET_RETURN = -21.88060747443149
            state_mean = np.array( [ 9.93873105e-01,  7.50027122e-01,  4.74504402e-01,  1.01736151e+00,
            7.48881207e-01,  4.06169418e-01,  2.34884081e-02, -1.14591538e-03,
            -6.83349842e-02, -4.13081264e-08,  2.98508486e-05,  2.14908690e-04,
            4.81348354e-03, -1.66446110e-01,  1.01778709e-03, -6.77563873e-05,
            -2.02580896e-03,  3.27181451e-04,  3.53621597e-04, -5.37996387e-03,
            -2.12234295e-04,  5.42253007e-05,  1.60631675e-03,  1.14111731e-03,
            1.15017950e-03,  1.01736151e+00,  7.48881207e-01,  4.06169418e-01,
            1.39622981e+00,  7.48640792e-01,  4.14018929e-01])
            state_std = np.array([8.57375211e-02, 9.00451027e-02, 5.85426801e-02, 1.47726484e-01,
            1.26179018e-01, 5.39706465e-02, 1.51742480e-01, 1.43489767e-01,
            8.04174020e-02, 1.28619109e-06, 2.07812745e-04, 2.49017679e-01,
            1.38566761e-01, 5.11880604e-01, 1.46817500e-02, 1.48188938e-02,
            1.46395473e-02, 4.90792198e-02, 4.94688978e-02, 2.98535334e-02,
            1.38808795e-02, 1.42285119e-02, 1.33669475e-02, 2.25115461e-03,
            2.25890346e-03, 1.47726484e-01, 1.26179018e-01, 5.39706465e-02,
            1.72864421e-01, 1.73098158e-01, 1.00000000e-06])

    if task == PUSH:
        ENV_NAME = 'FetchPushDense-v2'
        if data == ALLEXPERT:
            TARGET_RETURN = -3.8630065284043273
            state_mean = np.array([ 1.34716478e+00,  7.42978403e-01,  4.54412906e-01,  1.35098278e+00,
            7.47562570e-01,  4.23275696e-01,  3.81799808e-03,  4.58416751e-03,
            -3.11372090e-02, -3.06502754e-08,  2.53966511e-05, -1.04804955e-02,
            6.32725944e-02,  2.47517941e-02,  2.40121234e-04,  1.68442124e-04,
            -1.04214057e-03, -3.60405272e-04,  3.43785356e-03,  5.30257405e-04,
            -1.62422103e-04, -1.69373329e-04,  9.12592471e-04,  7.79478949e-04,
            7.22123290e-04,  1.35098278e+00,  7.47562570e-01,  4.23275696e-01,
            1.34769095e+00,  7.48332161e-01,  4.24699754e-01])

            state_std = np.array( [9.02965655e-02, 9.32832768e-02, 3.16258733e-02, 9.60164091e-02,
            9.29307267e-02, 2.84797738e-02, 7.91208005e-02, 7.77869986e-02,
            4.23918064e-02, 1.21235134e-06, 1.76953160e-04, 9.01434630e-01,
            4.35373235e-01, 7.38621687e-01, 1.17419569e-02, 1.20242885e-02,
            1.06726754e-02, 9.02013655e-02, 8.85027902e-02, 5.59211136e-02,
            1.20179672e-02, 1.22979315e-02, 1.01242312e-02, 1.61984423e-03,
            1.50983630e-03, 9.60164091e-02, 9.29307267e-02, 2.84797738e-02,
            8.65211748e-02, 8.66093355e-02, 1.00000000e-06])
            
        if data == ALLRANDOM:
            TARGET_RETURN = -8.586089636844338
            state_mean = np.array([ 1.33981335e+00,  7.48671111e-01,  4.76742511e-01,  1.34825854e+00,
            7.49052696e-01,  4.23782712e-01,  8.44519032e-03,  3.81584882e-04,
            -5.29597954e-02, -3.06502754e-08,  2.53966511e-05, -5.52413216e-04,
            1.16198198e-03,  9.85843659e-04,  3.58046071e-04, -7.27576844e-06,
            -1.68701133e-03, -3.14434777e-05,  1.51370498e-04,  3.55477059e-05,
            -3.53210222e-04,  1.10082550e-05,  1.62877541e-03,  1.16463639e-03,
            1.16406391e-03,  1.34825854e+00,  7.49052696e-01,  4.23782712e-01,
            1.34813351e+00,  7.49387441e-01,  4.24699754e-01])

            state_std = np.array( [8.21103906e-02, 8.83638422e-02, 5.95232258e-02, 1.04980306e-01,
            1.04443454e-01, 1.97978841e-02, 1.31418830e-01, 1.35429137e-01,
            6.27681580e-02, 1.21235134e-06, 1.76953160e-04, 2.22204647e-01,
            1.22503614e-01, 1.98596412e-01, 1.41076306e-02, 1.44709206e-02,
            1.38555791e-02, 2.11979274e-02, 2.21871918e-02, 1.55611697e-02,
            1.40943103e-02, 1.44684403e-02, 1.37041474e-02, 2.28013360e-03,
            2.27679976e-03, 1.04980306e-01, 1.04443454e-01, 1.97978841e-02,
            8.69310020e-02, 8.64701117e-02, 1.00000000e-06])

        if data == NINETYRANDOMTENEXPERT:
            TARGET_RETURN = -8.09569798924034
            state_mean = np.array([ 1.340449410e+0,  7.48324794e-01,  4.74511694e-01,  1.34825778e+00,
            7.48821001e-01,  4.23799104e-01,  7.80837113e-03,  4.96207240e-04,
            -5.07125868e-02, -3.06502754e-08,  2.53966511e-05, -1.27213318e-04,
            6.86366874e-03,  2.39932671e-03,  3.45535146e-04,  3.92344760e-07,
            -1.62118255e-03, -2.91798627e-05,  4.35913436e-04,  4.93372707e-05,
            -3.34074562e-04, -2.16944889e-06,  1.55790770e-03,  1.12385067e-03,
            1.11936841e-03,  1.34825778e+00,  7.48821001e-01,  4.23799104e-01,
            1.34790602e+00,  7.49156179e-01,  4.24699754e-01])

            state_std = np.array([8.27094440e-02, 8.86621772e-02, 5.77163900e-02, 1.03913216e-01,
            1.03248761e-01, 2.01795946e-02, 1.27084141e-01, 1.30787942e-01,
            6.11834947e-02, 1.21235134e-06, 1.76953160e-04, 3.45400779e-01,
            1.77408612e-01, 2.89835399e-01, 1.38873425e-02, 1.42406021e-02,
            1.35658514e-02, 3.45529273e-02, 3.43375351e-02, 2.27900945e-02,
            1.38976593e-02, 1.42604511e-02, 1.33907546e-02, 2.22338676e-03,
            2.21458856e-03, 1.03913216e-01, 1.03248761e-01, 2.01795946e-02,
            8.68339388e-02, 8.65287546e-02, 1.00000000e-06])
            

    if task == REACH:
        ENV_NAME = 'FetchReachDense-v2'
        if data == ALLEXPERT:
            TARGET_RETURN = -2.029989355941465
            state_mean = np.array([ 1.33927880e+00,  7.47559178e-01,  5.37941606e-01,  4.03683938e-06,
            1.45957759e-06, -6.93719664e-05, -3.12199700e-05,  1.22898961e-04,
            5.10460303e-04,  4.91192074e-04,  1.33927880e+00,  7.47559178e-01,
            5.37941606e-01,  1.34160574e+00,  7.48000767e-01, 5.36822742e-01])

            state_std = np.array([8.00639841e-02, 7.92910458e-02, 7.75973354e-02, 2.89680436e-05,
            1.11122502e-05, 1.04722242e-02, 1.05871841e-02, 1.04279512e-02,
            1.13095136e-03, 1.08557198e-03, 8.00639841e-02, 7.92910458e-02,
            7.75973354e-02, 8.66756833e-02, 8.72985199e-02, 8.67926843e-02])
            

        if data == ALLRANDOM:
            TARGET_RETURN = -9.515499597992534
            state_mean = np.array([ 1.33405742e+00,  7.49179725e-01,  5.40227604e-01,  4.03683938e-06,
            1.45957759e-06, -3.01273195e-04,  9.26788096e-06,  2.87770290e-04,
            6.36248515e-04,  6.34441293e-04,  1.33405742e+00,  7.49179725e-01,
            5.40227604e-01,  1.34254303e+00,  7.48941670e-01,  5.34054588e-01])

            state_std = np.array( [8.38169964e-02, 9.16313342e-02, 7.70230398e-02, 2.89680436e-05,
            1.11122502e-05, 1.44805361e-02, 1.48264651e-02, 1.47521653e-02,
            1.21473132e-03, 1.20942335e-03, 8.38169964e-02, 9.16313342e-02,
            7.70230398e-02, 8.65121104e-02, 8.63020567e-02, 8.68003884e-02])


        if data == NINETYRANDOMTENEXPERT:
            TARGET_RETURN =  -8.760961617933884
            state_mean = np.array([ 1.33427046e+00,  7.49169808e-01,  5.39960504e-01,  4.03683938e-06,
            1.45957759e-06, -2.83272730e-04,  2.87692183e-06,  2.76966278e-04,
            6.23581645e-04,  6.18648457e-04,  1.33427046e+00,  7.49169808e-01,
            5.39960504e-01,  1.34311568e+00,  7.48870778e-01,  5.34598790e-01])

            state_std = np.array( [8.35862754e-02, 9.03810986e-02, 7.70036388e-02, 2.89680436e-05,
            1.11122502e-05, 1.41402012e-02, 1.44458239e-02, 1.43736951e-02,
            1.20799581e-03, 1.19164978e-03, 8.35862754e-02, 9.03810986e-02,
            7.70036388e-02, 8.62971624e-02, 8.63469454e-02, 8.70359140e-02])


    
    env = env = gym.make(ENV_NAME, render_mode='rgb_array')
    model = DecisionTransformerModel.from_pretrained(modelpath)
    state_dim = np.array(np.sum([env.observation_space['observation'].shape[0], env.observation_space['achieved_goal'].shape[0], env.observation_space['desired_goal'].shape[0]]))# state size
    act_dim = env.action_space.shape[0] # action size


    state_mean = torch.from_numpy(state_mean)
    state_std = torch.from_numpy(state_std)
    scale = 50

    runs_frames = []
    runs_actions = None
    runs_rewards = None
    runs_states = None
    runs_timesteps = None
    runs_returns = None
    runs_successful = 0


    for i in range(num_runs):

        frames = []
        obs = env.reset()[0]
        state = np.concatenate([obs['observation'], obs['achieved_goal'], obs['desired_goal']])
        frames.append(env.render())
        target_return = torch.tensor(TARGET_RETURN).float().reshape(1, 1)
        states = torch.from_numpy(state).reshape(1, state_dim).float()
        actions = torch.zeros((0, act_dim)).float()
        rewards = torch.zeros(0).float()
        timesteps = torch.tensor(0).reshape(1, 1).long()

        success = False

        # take steps in the environment
        for t in range(MAX_EPISODE_LENGTH):
            # add zeros for actions as input for the current time-step
            actions = torch.cat([actions, torch.zeros((1, act_dim))], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1)])

            # predicting the action to take
            action = get_action(model,
                                state_dim,
                                act_dim,
                                (states - state_mean) / state_std,
                                actions,
                                rewards,
                                target_return,
                                timesteps)
            
            actions[-1] = action
            action = action.detach().numpy()

            # interact with the environment based on this action
            obs, reward, done, _, _ = env.step(action)
            state = np.concatenate([obs['observation'], obs['achieved_goal'], obs['desired_goal']])
            frames.append(env.render())

            if reward >= -0.05:
                success == True
            
            cur_state = torch.from_numpy(state).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward
            
            pred_return = target_return[0, -1] - (reward / scale)
            pred_return = target_return[0, -1] - (reward)
            target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((1, 1)).long() * (t + 1)], dim=1)
            
            if done:
                break

        runs_frames.append(frames)
        if success:
            runs_successful += 1

        # runs_actions.append(actions)
        # concat the actions from this run to the overall actions should be shape (num_runs, num_timesteps, act_dim)
        actions_t = actions.reshape(1, -1, act_dim)
        target_return_t = target_return.reshape(1, -1, 1)
        states_t = states.reshape(1, -1, state_dim)
        rewards_t = rewards.reshape(1, -1)
        if runs_actions == None:
            runs_actions = actions_t
            runs_returns = target_return_t
            runs_states = states_t
            runs_timesteps = timesteps
            run_rewards = rewards_t
        else:
            runs_actions = torch.cat([runs_actions, actions_t], dim=0)
            runs_returns = torch.cat([runs_returns, target_return_t], dim=0)
            runs_states = torch.cat([runs_states, states_t], dim=0)
            runs_timesteps = torch.cat([runs_timesteps, timesteps], dim=0)
            run_rewards = torch.cat([run_rewards, rewards_t], dim=0)
            
    return runs_successful, run_rewards, runs_frames, runs_actions, runs_returns, runs_states, runs_timesteps

def parseargs():
    parser = argparse.ArgumentParser(description="Decision Transformer for Robotic Control")
    parser.add_argument('-e' '--environment', type=str, required=True, dest='environment', help="Which environment to train. Options are [Pick, Push, Reach, Slide]")
    parser.add_argument('-m' '--modelspath', type=str, required=True, dest='modelspath', help="")
    parser.add_argument('-f' '--dirpath', type=str, required=True, dest='dirpath', help="")
    parser.add_argument('-d', '--data', type=str, required=True, dest='data', help="")
    parser.add_argument('-nr', '--num_runs', type=int, required=False, dest='num_runs', default=1, help="")
    parser.add_argument('-l', '--max_episode_length', type=int, required=False, dest='max_episode_length', default=50, help="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()

    task = args.environment
    modelspath = args.modelspath
    dir_filepath = args.dirpath
    data = args.data
    num_runs = args.num_runs
    max_episode_length = args.max_episode_length


    ask = 'y'
    if os.path.exists(dir_filepath):
        print("Directory already exists")
        ask = input("Do you want to overwrite? (y/n)")
        if ask.lower() == 'y' or ask.lower() == 'yes':
            shutil.rmtree(dir_filepath)
        else:
            exit()

    os.mkdir(dir_filepath)

    models = []
    for root, dirs, files in os.walk(modelspath):
        for d in dirs:
            models.append(os.path.join(root, d))
    
    reward_means = []
    reward_stds = []
    successful = []
    for idx, model in enumerate(models):
        if idx%10 == 0 or idx == len(models)-1:
            print(idx)
            runs_successful, run_rewards, runs_frames, runs_actions, runs_returns, runs_states, runs_timesteps = inference(model, task, data, num_runs, MAX_EPISODE_LENGTH=max_episode_length)
            successful.append(runs_successful)
            run_rewards = torch.sum(run_rewards, axis=1)
            reward_means.append(run_rewards.mean())
            reward_stds.append(run_rewards.std())
    

    np.save(os.path.join(dir_filepath, 'reward_means.npy'), np.array(reward_means))
    np.save(os.path.join(dir_filepath, 'reward_stds.npy'), np.array(reward_stds))
    np.save(os.path.join(dir_filepath, 'successful.npy'), np.array(successful))

