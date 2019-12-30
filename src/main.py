import os
from DQNAgent import *
from tqdm import tqdm
from snake import SnakeEnv

# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 0.1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

# Stats settings
AGGREGATE_STATS_EVERY = 1
SAVE_MODELS = False

# Render
SHOW_PREVIEW = True

# For stats
ep_rewards = [-200]

# For more repetitive results
# random.seed(1)
# np.random.seed(1)
# tf.random.set_seed(1)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

env = SnakeEnv()
agent = DQNAgent(env)
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episode"):
    agent.tensorboard.step = episode
    episode_reward = 0
    step = 1
    current_state = env.reset()

    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)
        episode_reward += reward

        if SHOW_PREVIEW and episode % AGGREGATE_STATS_EVERY == 0:
            env.render()

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done)

        current_state = new_state
        step += 1

    ep_rewards.append(episode_reward)
    if SAVE_MODELS and episode % AGGREGATE_STATS_EVERY == 0:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.model.save(f'models/{MODEL_NAME}_{max_reward:_>7.2f}'
                         f'max_{average_reward:_>7.2f}'
                         f'avg_{min_reward:_>7.2f}'
                         f'min.model')

    print(f"  Episode: {episode}   Score: {env.score}")

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
