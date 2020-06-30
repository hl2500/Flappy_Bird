import numpy as np
import paddle.fluid as fluid
import parl
from parl import layers

class FlappyBirdAgent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 e_greed=0.1,
                 e_greed_decrement=0):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(FlappyBirdAgent, self).__init__(algorithm)

        self.global_step = 0
        self.update_target_steps = 200  # copy parameters from model to target model in every 200 training steps

        self.e_greed = e_greed  # probability to exploring actions randomly
        self.e_greed_decrement = e_greed_decrement  # as training gradually converges, less exploration can be implemented

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program): # build computational graph to predict actions, define input/output variables
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program): # build computational graph to update Q network, define input/output variables
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal)

    def sample(self, obs):
        sample = np.random.rand() # generate random number between 0-1
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim) # exploration: every action may be selected
        else:
            act = self.predict(obs) # choose the best action
        self.e_greed = max(0.15, self.e_greed - self.e_greed_decrement) # as training gradually converges, less exploration can be implemented
        return act

    def predict(self, obs):  # choose the best action
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)  # choose the index of max Q value: the corresponding action
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        # copy parameters from model to target model in every 200 training steps
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal,
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]  # train the network once
        return cost

