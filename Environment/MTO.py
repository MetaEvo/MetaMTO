import copy
import numpy as np
from Environment.operators import DE_rand_1, DE_ran_1_bin, DE_ran_2_bin, DE_best_1_bin, DE_best_2_bin

INT_MAX = 1e+32
class Instance():
    def __init__(self, pop_cnt, dim, task ,total_generation):
        self.pop_cnt = pop_cnt
        self.dim = dim
        self.task= task

        self.begin_best = 0
        self.begin_worst = 0
        self.gbest = INT_MAX
        self.last_gen_best = 0
        self.this_gen_best = 0

        self.generation = 0
        self.total_generation = total_generation
        self.flag_improved = 0
        self.stagnation = 0
        self.N_kt = 0
        self.Q_kt = 0

        self.offsprings = None
        self.noKT_offsprings = None
        self.KT_offsprings = None
        self.KT_index = None
        self.parent_population = np.array([np.random.rand(dim) for i in range(pop_cnt)])
        self.parent_fitness = None
        self.best_individual = None

        self.reward = 0
        self.converge_reward = 0
        self.kt_reward = 0

        self.best_fitness_track = np.array([INT_MAX for i in range(self.total_generation)], dtype=np.float32)


        fitness_list = []
        for i in range(pop_cnt):
            fitness = self.evaluation(self.parent_population[i])
            fitness_list.append(fitness)
            if fitness < self.gbest:
                self.gbest = min(fitness, self.gbest)
                self.best_individual = self.parent_population[i]

            if fitness > self.begin_worst:
                self.begin_worst = max(fitness, self.begin_worst)

        self.begin_best = self.gbest
        self.parent_fitness = np.array(fitness_list)

        self.last_gen_best = self.gbest
        self.this_gen_best = self.gbest
        self.best_fitness_track[self.generation] = self.gbest
        self.optimal_value = self.task.optimum


    def evaluation(self, indiv):
        solution = self.task.decode(indiv)
        cost = self.task.func(solution)

        return cost

    def get_state(self):
        states = []
        state_1 = self.stagnation / self.total_generation
        states.append(state_1)
        state_2 = self.flag_improved
        states.append(state_2)
        if self.N_kt == 0:
            state_3 = 0
        else:
            state_3 = self.Q_kt

        states.append(state_3)
        state_4 = np.mean(np.std(self.parent_population, axis=0))
        states.append(state_4)

        state_parent_fitness = (self.parent_fitness - self.optimal_value) / (self.begin_worst - self.optimal_value)
        state_5 = np.std(state_parent_fitness)
        states.append(state_5)

        self.generation += 1
        return np.array(states, dtype=np.float32)


    def update(self):
        self.noKT_offsprings = DE_rand_1(self.parent_population)

    def transfer(self, source_population, source_best_indiv, action1, action2, action3, action4):
        action_1 = action1
        action_2 = action2
        action_3 = action3
        action_4 = action4

        self.N_kt = 0.5 * action_1.cpu()
        self.KT_count = int(np.ceil(self.N_kt * self.pop_cnt))
        if self.KT_count == 0:
            self.KT_count = 1
        self.KT_index = np.random.choice(np.arange(self.pop_cnt), size=self.KT_count, replace=False)

        F = action_3.cpu().numpy()
        CR = action_4.cpu().numpy()
        if action_2 == 0:
            self.KT_offsprings = DE_ran_1_bin(self.parent_population, source_population, self.KT_index, F, CR)
        if action_2 == 1:
            self.KT_offsprings = DE_best_1_bin(self.parent_population, self.best_individual, source_population, self.KT_index, F, CR)
        if action_2 == 2:
            self.KT_offsprings = DE_ran_2_bin(self.parent_population, source_population, self.KT_index, F, CR)
        if action_2 == 3:
            self.KT_offsprings = DE_best_2_bin(self.parent_population, source_population, source_best_indiv, self.KT_index, F, CR)

        self.offsprings = copy.deepcopy(self.noKT_offsprings)
        for i in range(self.KT_count):
            self.offsprings[self.KT_index[i]] = self.KT_offsprings[i]

    def selection(self):
        self.last_gen_best = self.this_gen_best
        parent_population_fitness = []
        offsprings_population_fitness = []
        for i in range(self.pop_cnt):
            parent_fitness = self.evaluation(self.parent_population[i])
            parent_population_fitness.append(parent_fitness)
            offspring_fitness = self.evaluation(self.offsprings[i])
            offsprings_population_fitness.append(offspring_fitness)

        parent_population_fitness = np.array(parent_population_fitness)
        offsprings_population_fitness = np.array(offsprings_population_fitness)
        next_population = copy.deepcopy(self.parent_population)

        S_update = 0
        S_KT = 0
        for i in range(self.pop_cnt):
            if offsprings_population_fitness[i] <= parent_population_fitness[i]:
                if i not in self.KT_index:
                    S_update += 1
                else:
                    S_KT += 1

                next_population[i] = self.offsprings[i]
            else:
                next_population[i] = self.parent_population[i]

        flag = 0
        for i in range(self.pop_cnt):
            fitness = self.evaluation(next_population[i])
            self.parent_fitness[i] = fitness
            if fitness < self.gbest:
                self.gbest = fitness
                self.best_individual = next_population[i]
                flag = 1
        if(flag):
            self.flag_improved = 1
        else:
            self.flag_improved = 0
            self.stagnation += 1

        self.this_gen_best = self.gbest
        self.parent_population = next_population
        self.best_fitness_track[self.generation] = self.gbest

        reward_KT = float(S_KT) / self.KT_count
        reward_converge = (self.last_gen_best - self.this_gen_best) / (self.begin_best - self.optimal_value)

        self.kt_reward = 0.5 * reward_KT
        self.converge_reward = 0.5 * reward_converge

        self.reward = self.kt_reward + self.converge_reward
        self.Q_kt = float(S_KT) / self.KT_count

class MTO_environment():
    def __init__(self, instances):
        self.instances = instances
        self.instance_cnt = len(instances)
        self.instance_best_result = np.array([INT_MAX for i in range(self.instance_cnt)], dtype=np.float32)

    def reset(self):
        states = []
        for i in range(self.instance_cnt):
            state = self.instances[i].get_state()
            states.append(state)
            self.instance_best_result[i] = self.instances[i].gbest
        return np.array(states, dtype=np.float32)

    def step(self, action):
        next_states = []
        total_reward = 0
        total_kt_reward = 0
        total_converge_reward = 0
        for i in range(self.instance_cnt):
            self.instances[i].update()
            rand_source_index = action[i][0]
            rand_source_instance = self.instances[rand_source_index]

            self.instances[i].transfer(rand_source_instance.parent_population, rand_source_instance.best_individual, action[i][1], action[i][2], action[i][3], action[i][4])
            self.instances[i].selection()
            state = self.instances[i].get_state()
            next_states.append(state)
            total_reward += self.instances[i].reward
            total_kt_reward += self.instances[i].kt_reward
            total_converge_reward += self.instances[i].converge_reward
            self.instance_best_result[i] = self.instances[i].gbest

        return np.array(next_states,dtype=np.float32), total_reward, total_kt_reward, total_converge_reward
