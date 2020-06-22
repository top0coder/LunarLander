# ENV_NAME = 'CartPole-v0'
ENV_NAME = 'LunarLander-v2'
EPISODES = 256
STEPS = 1024
MEMORY_SIZE = 1000000
BATCH_SIZE = 64
MEMORY_WARMUP_SIZE = 128
LEARNING_RATE = 1e-3
GAMMA = 0.98
LEARN_FREQ = 1
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.98
SAVE_PATH = './ckpts/'
RUN_TAG = 'run_1'
SAVE_FREQ = 20