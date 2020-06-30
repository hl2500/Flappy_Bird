import paddle.fluid as fluid
import parl
from parl import layers

class FlappyBirdModel(parl.Model):
    def __init__(self, act_dim):
        hid1_size = 256
        hid2_size = 128
        # 3 fully connected layers
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act=None)

    def value(self, obs):
        # input the state and output the corresponding Q values ([Q(s,a1), Q(s,a2), Q(s,a3)...]) of all actions
        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        Q = self.fc3(h2)
        return Q
