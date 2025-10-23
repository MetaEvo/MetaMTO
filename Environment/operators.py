import copy
import numpy as np

def DE_mutation(populations):
    F = 0.5
    pop_cnt, dim = populations.shape
    mutants = copy.deepcopy(populations)
    for j in range(pop_cnt):
        r1 = np.random.randint(low=0, high=pop_cnt)
        r2 = np.random.randint(low=0, high=pop_cnt)
        r3 = np.random.randint(low=0, high=pop_cnt)
        while r1 == j:
            r1 = np.random.randint(low=0, high=pop_cnt)
        while r2 == r1 or r2 == j:
            r2 = np.random.randint(low=0, high=pop_cnt)
        while r3 == r2 or r3 == r1 or r3 == j:
            r3 = np.random.randint(low=0, high=pop_cnt)

        x1 = populations[r1]
        x2 = populations[r2]
        x3 = populations[r3]
        mutant = x1 + F * (x2 - x3)
        mutant = np.clip(mutant, a_min=0, a_max=1)
        mutants[j] = mutant

    return mutants

def DE_crossover(mutants, populations, CR=0.7):
    U = copy.deepcopy(mutants)
    population_cnt, dim = mutants.shape
    for j in range(population_cnt):
        rand_pos = np.random.randint(low=0, high=dim)
        for k in range(dim):
            mutant = mutants[j]
            rand = np.random.rand()
            if rand <= CR or k == rand_pos:
                U[j][k] = mutant[k]

            if rand > CR and k != rand_pos:
                U[j][k] = populations[j][k]
    return U

def DE_rand_1(populations):
    mutants = DE_mutation(populations)
    DE_offsprings = DE_crossover(mutants, populations)
    return DE_offsprings


def DE_ran_1_bin(populations, source_populations, KT_index, F, CR):
    population_target = populations[KT_index]
    pop_cnt, dim = source_populations.shape
    mutants = []
    for i in range(population_target.shape[0]):
        r1, r2, r3= np.random.choice(np.arange(pop_cnt), size=3, replace=False)
        X_r1 = populations[r1]
        X_r2 = source_populations[r2]
        X_r3 = source_populations[r3]

        mutant = X_r1 + F * (X_r2 - X_r3)
        mutant = np.clip(mutant, a_min=0, a_max=1)
        mutants.append(mutant)

    mutants = np.array(mutants)
    U = DE_crossover(mutants, population_target, CR)

    return U

def DE_ran_2_bin(populations, source_populations, KT_index, F, CR):
    population_target = populations[KT_index]
    pop_cnt, dim = source_populations.shape
    mutants = []
    for i in range(population_target.shape[0]):
        r1, r2, r3 = np.random.choice(np.arange(pop_cnt), size=3, replace=False)
        X_r1 = source_populations[r1]
        X_r2 = populations[r2]
        X_r3 = populations[r3]

        mutant = X_r1 + F * (X_r2 - X_r3)
        mutant = np.clip(mutant, a_min=0, a_max=1)
        mutants.append(mutant)

    mutants = np.array(mutants)
    U = DE_crossover(mutants, population_target, CR)

    return U

def DE_best_1_bin(populations, target_best_indiv, source_populations, KT_index, F, CR):
    population_target = populations[KT_index]
    pop_cnt, dim = source_populations.shape
    mutants = []
    for i in range(population_target.shape[0]):
        r1, r2, r3 = np.random.choice(np.arange(pop_cnt), size=3, replace=False)
        X_r1 = target_best_indiv
        X_r2 = source_populations[r2]
        X_r3 = source_populations[r3]

        mutant = X_r1 + F * (X_r2 - X_r3)
        mutant = np.clip(mutant, a_min=0, a_max=1)
        mutants.append(mutant)

    mutants = np.array(mutants)
    U = DE_crossover(mutants, population_target, CR)

    return U

def DE_best_2_bin(populations, source_populations, source_best_indiv, KT_index, F, CR):
    population_target = populations[KT_index]
    pop_cnt, dim = source_populations.shape
    mutants = []
    for i in range(population_target.shape[0]):
        r1, r2, r3 = np.random.choice(np.arange(pop_cnt), size=3, replace=False)
        X_r1 = source_best_indiv
        X_r2 = populations[r2]
        X_r3 = populations[r3]

        mutant = X_r1 + F * (X_r2 - X_r3)
        mutant = np.clip(mutant, a_min=0, a_max=1)
        mutants.append(mutant)

    mutants = np.array(mutants)
    U = DE_crossover(mutants, population_target, CR)

    return U

