
import os, sys
import gym
import numpy as np
import parl
from parl.utils import logger 
from parl.algorithms import DQN  # parl >= 1.3.1

from agent import Agent
from replay_memory import ReplayMemory
from model import Model
from config import *

def run_episode(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    for _ in range(STEPS):
        step += 1
        action = agent.sample(obs)
        next_obs, reward, done, _ = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # s,a,r,s',done

        total_reward += reward
        obs = next_obs
        env.render()
        if done:
            break
    return total_reward


def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs) # only select the best
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def train():
    env = gym.make(ENV_NAME)
    env.seed(0)
    np.random.seed(0)

    action_dim = env.action_space.n  
    obs_shape = env.observation_space.shape

    rpm = ReplayMemory(MEMORY_SIZE)

    model = Model(act_dim=action_dim)
    algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm,
        obs_dim=obs_shape[0],
        act_dim=action_dim)

    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(env, agent, rpm)

    episode = 0
    rewards = []
    for episode in range(EPISODES):
        total_reward = run_episode(env, agent, rpm)
        episode += 1
        rewards.append(total_reward)
        average_reward = np.mean(rewards[-100:])
        logger.info(f'episode:{episode}\t e_greed:{agent.epsilon:6f}\t reward:{total_reward:.2f} \t'
                    f'Avg reward: {average_reward:.2f}\n')
        if (episode+1) % SAVE_FREQ:
            agent.save(os.path.join(SAVE_PATH, f'{RUN_TAG}_{episode+1}.ckpt'))

    agent.save(os.path.join(SAVE_PATH, f'{RUN_TAG}_final.ckpt'))

def play():
    env = gym.make(ENV_NAME)
    action_dim = env.action_space.n  
    obs_shape = env.observation_space.shape

    model = Model(act_dim=action_dim)
    algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm,
        obs_dim=obs_shape[0],
        act_dim=action_dim)

    ckpt_path = './dqn_model.ckpt'
    agent.restore(ckpt_path)

    eval_reward = evaluate(env, agent, render=True)  # render=True 查看显示效果
    logger.info(f'Avg test reward:{eval_reward:.2f}')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('wrong number of arguments')
        sys.exit()
    cmd = sys.argv[1]
    if cmd == 'train':
        train()
    elif cmd == 'play':
        play()
    else:
        print('unk cmd')

