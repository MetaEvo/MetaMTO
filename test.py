from Environment.MTO import MTO_environment, Instance
from agent import Meta_MTO_agent
import numpy as np
import os
import torch
from config import get_config
import tqdm
import pickle
import matplotlib.pyplot as plt
import ast


def test_result(agent, tasks, n_steps=250):
    instances = [Instance(50, DIM, tasks[i], n_steps) for i in range(TASK_CNT)]
    env = MTO_environment(instances)
    total_converge_reward = 0
    state = env.reset()
    for t in range(1, n_steps):
        action, fixed_action, logprobs = agent.get_action(torch.tensor(state).unsqueeze(0))
        next_state, reward, reward_kt, reward_converge = env.step(action)
        total_converge_reward += reward_converge
        state = next_state

    return env.instance_best_result, total_converge_reward

def plot_result():
    filename = 'converge_reward.txt'
    benchmarks = []

    with open(filename, 'r') as file:
        for line in file:
            key, value = line.split(':', 1)
            benchmark_values = ast.literal_eval(value.strip())
            benchmarks.append(benchmark_values)

    benchmarks = np.array(benchmarks)
    mean_value = np.mean(benchmarks, axis=0)
    data = mean_value
    x = list(range(len(data)))

    plt.plot(x, data, marker='o', linestyle='-', color='b', label='Data Line')
    plt.title('Reward')
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    np.random.seed(2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = get_config()
    n_state = config.n_state
    DIM = config.dim
    TASK_CNT = config.task_cnt
    load_name = config.load_name

    model_load_dir = os.path.join('saved_models', load_name)

    dataset_path = os.path.join("Augmented_WCCI_Dataset_Test", f"{config.difficulty}.pkl")
    with open(dataset_path, "rb") as f:
        tasks_set = pickle.load(f)

    task_choice = np.random.choice(np.arange(0,127),size=(60,),replace=False)

    converge_reward_list = [[] for _ in range(len(task_choice))]

    pbar = tqdm.tqdm(total=int((config.episode+1)/8+1), desc=f'test_episode', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

    for i in range(0, config.episode+1, 8):
        agent = Meta_MTO_agent(n_state, 64, 64).to(device)
        if i==0:
            path = os.path.join(model_load_dir, f'episode_init')
            agent.load_state_dict(torch.load(path))
        if i > 0:
            path = os.path.join(model_load_dir, f'episode_{i-1}')
            agent.load_state_dict(torch.load(path))
        for task_id in range(len(task_choice)):
            tasks = tasks_set[task_choice[task_id]]
            result, total_converge_reward = test_result(agent, tasks)
            converge_reward_list[task_id].append(total_converge_reward)

        pbar.update()
    pbar.close()

    with open("converge_reward.txt","a") as f:
        for task_id in range(len(task_choice)):
            f.writelines(f"benchmark_{task_choice[task_id]}: ")
            f.writelines(str(converge_reward_list[task_id]))
            f.write("\n")

    plot_result()
