import torch
import numpy as np
from Environment.MTO import MTO_environment, Instance
from agent import Meta_MTO_agent
import os
import tqdm
import pickle
from tensorboard_logger import Logger
import time

def seed_settings(config):
    seed_dict = {'VS':19, 'S':13, 'M':16, 'L':18, 'VL':22}
    seed = seed_dict[config.difficulty]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train(n_steps,config):
    seed_settings(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_state = config.n_state
    DIM = config.dim
    TASK_CNT = config.task_cnt
    run_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    logdir = os.path.join('log', run_name)
    logger = Logger(logdir)
    episode_model_save_dir = os.path.join('saved_models', run_name)
    agent = Meta_MTO_agent(n_state, 64, 64).to(device)

    if not os.path.exists(episode_model_save_dir):
        os.makedirs(episode_model_save_dir)
    torch.save(agent.state_dict(), os.path.join(episode_model_save_dir, f'episode_init'))

    dataset_path = os.path.join("Augmented_WCCI_Dataset_Train", f"{config.difficulty}.pkl")
    with open(dataset_path, "rb") as f:
        tasks_set = pickle.load(f)

    pbar = tqdm.tqdm(total=len(tasks_set), desc=f'train_epoch', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

    for episode in range(len(tasks_set)):
        if episode > 0:
            path = os.path.join(episode_model_save_dir, f'episode_{episode-1}')
            agent.load_state_dict(torch.load(path))

        tasks = tasks_set[episode]
        instances = [Instance(50, DIM, tasks[i], n_steps) for i in range(TASK_CNT)]
        env = MTO_environment(instances)
        total_reward = 0
        total_converge_reward = 0
        transition_dict = {'states': [], 'logprobs': [], 'next_states': [], 'rewards': [], 'action': []}
        state = env.reset()
        for t in range(1, n_steps):
            action, fixed_action, logprobs = agent.get_action(torch.tensor(state).unsqueeze(0))
            next_state, reward, reward_kt, reward_converge = env.step(action)
            transition_dict['states'].append(state)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['logprobs'].append(logprobs)
            transition_dict['action'].append(fixed_action)
            total_reward += reward
            total_converge_reward += reward_converge
            state = next_state

            if (t % 10 == 0 and t != 0):
                agent.update(transition_dict)
                transition_dict = {'states': [], 'logprobs': [], 'next_states': [], 'rewards': [], 'action': []}

        logger.log_value('total_reward', total_reward, episode)
        logger.log_value('total_converge_reward', total_converge_reward, episode)

        if not os.path.exists(episode_model_save_dir):
            os.makedirs(episode_model_save_dir)
        torch.save(agent.state_dict(), os.path.join(episode_model_save_dir, f'episode_{episode}'))

        pbar.update()
    pbar.close()