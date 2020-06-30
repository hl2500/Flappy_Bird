import numpy as np
import parl
from parl.utils import logger

from flappy_bird_model import FlappyBirdModel
from flappy_bird_agent import FlappyBirdAgent
from replay_memory import ReplayMemory

from ple.games.flappybird import FlappyBird
from ple import PLE

LEARN_FREQ = 5  # update parameters every 5 steps
MEMORY_SIZE = 20000  # replay memory size
MEMORY_WARMUP_SIZE = 200  # store some experiences in the replay memory in advance
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.99  # discount factor of reward

def run_episode(agent, env, rpm):

    total_reward = 0
    step = 0
    env.init()
    obs = list(env.getGameState().values())

    while True:
        step += 1
        action = agent.sample(obs)  # sample actions
        reward = env.act(env.getActionSet()[action])
        next_obs = list(env.getGameState().values())
        done = env.game_over()
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)

        total_reward += reward
        obs = next_obs
        if done:
            env.reset_game()
            break

    return total_reward

# evaluate agent, total reward is the mean of 5 episodes
def evaluate(agent, env):

    eval_reward = []

    for i in range(5):
        env.init()
        obs = list(env.getGameState().values())
        episode_reward = 0

        while True:
            action = agent.predict(obs)
            reward = env.act(env.getActionSet()[action])
            obs = list(env.getGameState().values())
            done = env.game_over()
            score = int(env.score())
            episode_reward += reward
            
            if done:
                env.reset_game()
                break
            if score > 500:
                print(f'The score is {score}, so start over')
                env.reset_game()
                break
 
        eval_reward.append(episode_reward)

    return np.mean(eval_reward)

def main():

    game = FlappyBird()
    env = PLE(game, fps=30, display_screen=False)
    env.init()

    action_dim = len(env.getActionSet())
    obs_shape = len(env.getGameState())

    rpm = ReplayMemory(MEMORY_SIZE)

    model = FlappyBirdModel(act_dim=action_dim)
    algorithm = parl.algorithms.DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = FlappyBirdAgent(
        algorithm,
        obs_dim=obs_shape,
        act_dim=action_dim,
        e_greed=0.2,
        e_greed_decrement=1e-6
    )  # probability of exploring is decreasing during training

    while len(rpm) < MEMORY_WARMUP_SIZE:  # warm up replay memory
        run_episode(agent, env, rpm)

    max_episode = 50000

    # start train
    episode = 0
    while episode < max_episode:

        # train part
        for i in range(0, 100):
            total_reward = run_episode(agent, env, rpm)
            episode += 1
        # evaluation part
        eval_reward = evaluate(agent, env)
        logger.info('episode:{}    test_reward:{}'.format(episode, eval_reward))
        # learning rate adjustment
        if episode % 100 == 0:
            if algorithm.lr >= 5e-4:
                algorithm.lr *= 0.995
            if algorithm.lr <= 5e-4 and algorithm.lr >= 1e-4:
                algorithm.lr *= 0.99
            print('learning rate:', algorithm.lr)

    # save model
    save_path = './fb_dqn_model.ckpt'
    agent.save(save_path)

if __name__ == '__main__':
    main()
