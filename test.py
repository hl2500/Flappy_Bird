import parl
from parl.utils import logger

import cv2
import numpy as np

from flappy_bird_model import FlappyBirdModel
from flappy_bird_agent import FlappyBirdAgent

from ple.games.flappybird import FlappyBird
from ple import PLE

LEARNING_RATE = 0.001
GAMMA = 0.99

game = FlappyBird()
env = PLE(game, fps=30, display_screen=False)

action_dim = len(env.getActionSet())
obs_shape = len(env.getGameState())

model = FlappyBirdModel(act_dim=action_dim)
algorithm = parl.algorithms.DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
agent = FlappyBirdAgent(
    algorithm,
    obs_dim=obs_shape,
    act_dim=action_dim,
    e_greed=0.2,
    e_greed_decrement=1e-6
)

# load model
save_path = './fb_dqn_model.ckpt'
agent.restore(save_path)

# evaluate agent, total reward is the mean of 5 episodes
def evaluate(agent):
    eval_reward = []
    for i in range(5):
        env.init()
        obs = list(env.getGameState().values())
        episode_reward = 0
        while True:
            # display the score
            score = int(env.score())
            picture = env.getScreenRGB()
            picture = cv2.transpose(picture)
            font = cv2.FONT_HERSHEY_TRIPLEX
            picture = cv2.putText(picture, str(score), (0, 25), font, 1, (255, 0, 0), 2)
            cv2.imshow("flappy_bird", picture)
            action = agent.predict(obs)
            reward = env.act(env.getActionSet()[action])
            obs = list(env.getGameState().values())
            done = env.game_over()
            episode_reward += reward
            if done:
                env.reset_game()
                break
        eval_reward.append(episode_reward)
        cv2.destroyAllWindows()
    return np.mean(eval_reward)

# It's show time!
eval_reward = evaluate(agent)
logger.info(f'test_reward:{eval_reward}')
