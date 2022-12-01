import sys
import random
import numpy as np

sys.path.append("/home/ustc/baojie/scheduler/sy3")

from utils import get_book, save_data

seeds = 999
random.seed(seeds)
np.random.seed(seeds)

size1, size2 = 5, 7
num_slots = 32
num_gps = 100
num_pre_booked = 750
num_population = 2000
num_generation = 1000
num_to_book = 50
to_book = get_book(num_to_book)

# prebooked table
initial_grid = np.zeros((num_slots, num_gps))
for i in range(num_pre_booked):
    x, y = np.random.randint(num_slots), np.random.randint(num_gps)
    while(initial_grid[x, y] == 1):
        x, y = np.random.randint(num_slots), np.random.randint(num_gps)
    initial_grid[x, y] = 1

# gennerate initial population
def initial_population(num_population=num_population):
    population = []
    for i in range(num_population):
        population.append([])
        for j in range(num_to_book):
            x, y = np.random.randint(num_slots), np.random.randint(num_gps)
            population[i].append((x, y))
    return np.array(population)

# position to binary code
def translate1(population):
    bin_population = np.zeros((num_population, num_to_book, size1+size2))
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            position = population[i, j]
            x, y = str(bin(position[0]))[2:], str(bin(position[1]))[2:]
            while (len(x) != size1) or (len(y) != size2):
                if len(x) != size1:
                    x = '0' + x
                if len(y) != size2:
                    y = '0' + y
                    
            z = x + y
            for k in range(len(z)):
                bin_population[i, j, k] = int(z[k])
    return bin_population

# binary code to position
def translate2(population):
    x_pop = population[:, :, 0:size1]
    y_pop = population[:, :, size1:(size1+size2)]

    x = x_pop.dot(2 ** np.arange(size1)[::-1])
    y = y_pop.dot(2 ** np.arange(size2)[::-1])

    position = []
    for i in range(x.shape[0]):
        position.append([])
        for j in range(x.shape[1]):
            position[i].append((x[i, j], y[i, j]))
    position = np.array(position, dtype = np.int32)
    return position

# calculate every solution fitness
def get_fitness(population):
    table = initial_grid.copy()
    positions = translate2(population)
    scores = []
    index_list = []
    for i in range(positions.shape[0]):
        score = 0
        index = 0
        for j in range(positions.shape[1]):
            k = 0
            position = positions[i, j]
            while k < to_book[j]:
                if ((position[0]+k) >= num_slots) or ((table[position[0]+k, position[1]]) == 1):
                    break
                k += 1
            
            if k == to_book[j]:
                index += 1
                l = 0
                while l < k:
                    table[position[0]+l, position[1]] = 1
                    l += 1

                up = max(0, position[0]-1)
                down = min(num_slots-1, position[0]+to_book[j])
                if table[up, position[1]] == 0:
                    if table[down, position[1]] == 0:
                        score += 0
                    else:
                        score += to_book[j]
                else:
                    if table[up,position[1]] == 0:
                        score += to_book[j]
                    else:
                        score += 2 * to_book[j]
        scores.append(score)
        index_list.append(index)
    return np.array(scores), index_list

def crossover_and_mutation(population, crossover_rate = 0.8):
    new_population = []
    for father in population:
        child = father # copy father gene
        if np.random.rand() < crossover_rate:
            mother = population[np.random.randint(num_population)] # choose mother gene
            cross_point = np.random.randint(0, size1+size2) # random cross point
            child[cross_point:] = mother[cross_point:]
        child = mutation(child)
        new_population.append(child)
    return np.array(new_population)

def mutation(child, mutation_rate=0.01):
    sum_y = 130
    if np.random.rand() < mutation_rate:
        while sum_y >= 100:
            child_cp = child.copy()
            mutate_point1 = np.random.randint(0, child_cp.shape[0])
            mutate_point2 = np.random.randint(0, child_cp.shape[1])
            child_cp[mutate_point1][mutate_point2] = 1 if (child_cp[mutate_point1][mutate_point2] == 0) else 0
            child_cp_y = child_cp[mutate_point1][size1:]
            temp = 0
            for k in child_cp_y:
                temp = 2*temp + k
            sum_y = temp

        child = child_cp
    return child

def select(populaion,fitness):
    idx = np.random.choice(np.arange(num_population), size= num_population, replace=True, p=fitness/fitness.sum())
    return populaion[idx]

def get_book_index(fit, ind):
    fit = fit.tolist()
    best_fit = max(fit)
    best_index = fit.index(best_fit)
    best_ind = ind[best_index]
    return best_ind

def main():
    init_pop = initial_population()
    population = translate1(init_pop)
    best_fitness = []
    ind_list = []
    for i in range(num_generation):
        population = crossover_and_mutation(population)
        fitness, indexs = get_fitness(population)
        # print(fitness)
        population = select(population, fitness)
        index = get_book_index(fitness, indexs)
        best_fitness.append(max(fitness))
        ind_list.append(index)
        if i % 100 == 0:
            print("generation: {} best fitness: {} index: {}".format(i, max(fitness), index))

    save_data(best_fitness, './fitness4.pickle')
    save_data(ind_list, './index4.pickle')

if __name__ == "__main__":
    main()
