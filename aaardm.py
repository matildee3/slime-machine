import requests
import numpy as np
import math
from mealpy import IntegerVar, SMA, BinaryVar
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.image import imread
from scipy.ndimage import gaussian_filter
from scipy.ndimage import sobel
from scipy.ndimage import gaussian_filter, uniform_filter
from functools import lru_cache
import numpy as np
from mealpy.optimizer import Optimizer
from matplotlib.colors import LinearSegmentedColormap


black = (31 / 255, 31 / 255, 31 / 255)  # RGB (31, 31, 31)
yellow = (246 / 255, 242 / 255, 108 / 255)  # RGB (246, 242, 108)
black_yellow_cmap = LinearSegmentedColormap.from_list(
    'black_yellow', [black, yellow], N=256)

def generate_random_numbers(api_key, num_numbers, min_value, max_value):
    # URL to the Random.org JSON-RPC API
    url = "https://api.random.org/json-rpc/4/invoke"

    # JSON payload for the generateIntegers method
    data = {
        "jsonrpc": "2.0",
        "method": "generateIntegers",
        "params": {
            "apiKey": api_key,
            "n": num_numbers,
            "min": min_value,
            "max": max_value,
            "replacement": True
        },
        "id": 1
    }

    # Making the POST request
    response = requests.post(url, json=data)

    # Check if the request was successful
    if response.status_code == 200:
        # Print the random numbers
        print("Random Numbers:", response.json()['result']['random']['data'])
    else:
        print("Failed to retrieve data:", response.status_code)



api_key = 'YOUR_API_KEY'
true_random_seed = generate_random_numbers(api_key, 1, 1, 1000000)  

np.random.seed(true_random_seed)  # the truly random seed



class DevSMA(Optimizer):


    def __init__(self, epoch: int = 10000, pop_size: int = 100, p_t: float = 0.03, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.p_t = self.validator.check_float("p_t", p_t, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "p_t"])
        self.sort_flag = True

    def initialize_variables(self):
        self.weights = np.zeros((self.pop_size, self.problem.n_dims))


    def evolve(self, epoch):

        # plus eps to avoid denominator zero
        ss = self.g_best.target.fitness - self.pop[-1].target.fitness + self.EPSILON
        # calculate the fitness weight of each slime mold
        for idx in range(0, self.pop_size):
            # Eq.(2.5)
            if idx <= int(self.pop_size / 2):
                self.weights[idx] = 1 + self.generator.uniform(0, 1, self.problem.n_dims) * \
                                    np.log10((self.g_best.target.fitness - self.pop[idx].target.fitness) / ss + 1)
            else:
                self.weights[idx] = 1 - self.generator.uniform(0, 1, self.problem.n_dims) * \
                                    np.log10((self.g_best.target.fitness - self.pop[idx].target.fitness) / ss + 1)
        a = np.arctanh(-(epoch / self.epoch) + 1)  # Eq.(2.4)
        b = 1 - epoch / self.epoch
        pop_new = []
        for idx in range(0, self.pop_size):
            # Update the Position of search agent
            if self.generator.uniform() < self.p_t:  # Eq.(2.7)
                pos_new = self.problem.generate_solution()
            else:
                p = np.tanh(np.abs(self.pop[idx].target.fitness - self.g_best.target.fitness))  # Eq.(2.2)
                vb = self.generator.uniform(-a, a, self.problem.n_dims)  # Eq.(2.3)
                vc = self.generator.uniform(-b, b, self.problem.n_dims)
                # two positions randomly selected from population, apply for the whole problem size instead of 1 variable
                id_a, id_b = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                pos_1 = self.g_best.solution + vb * (self.weights[idx] * self.pop[id_a].solution - self.pop[id_b].solution)
                pos_2 = vc * self.pop[idx].solution
                condition = self.generator.random(self.problem.n_dims) < p
                pos_new = np.where(condition, pos_1, pos_2)
            # Check bound and re-calculate fitness after each individual move
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        if epoch == 1 or epoch % 100 == 0:
            edge_map = self.g_best.solution.reshape(img.shape)
            plt.figure(figsize=(6, 6))
            #plt.imshow(edge_map, cmap='gray')
            plt.imshow(edge_map, cmap=black_yellow_cmap)
            plt.axis('off')
            plt.title(f'Epoch {epoch}')
            plt.savefig(f'final/epoch_{epoch:05d}.png')
            plt.close()  # Close the figure to free memory



class OriginalSMA(DevSMA):


    def __init__(self, epoch=10000, pop_size=100, p_t=0.03, **kwargs):
        super().__init__(epoch, pop_size, p_t, **kwargs)

    def evolve(self, epoch):
        # plus eps to avoid denominator zero
        ss = self.g_best.target.fitness - self.pop[-1].target.fitness + self.EPSILON
        # calculate the fitness weight of each slime mold
        for idx in range(0, self.pop_size):
            # Eq.(2.5)
            if idx <= int(self.pop_size / 2):
                self.weights[idx] = 1 + self.generator.uniform(0, 1, self.problem.n_dims) * \
                    np.log10((self.g_best.target.fitness - self.pop[idx].target.fitness) / ss + 1)
            else:
                self.weights[idx] = 1 - self.generator.uniform(0, 1, self.problem.n_dims) * \
                    np.log10((self.g_best.target.fitness - self.pop[idx].target.fitness) / ss + 1)

        aa = np.arctanh(-(epoch / self.epoch) + 1)  # Eq.(2.4)
        bb = 1 - epoch / self.epoch
        pop_new = []
        for idx in range(0, self.pop_size):
            # Update the Position of search agent
            pos_new = self.pop[idx].solution.copy()
            if self.generator.uniform() < self.p_t:  # Eq.(2.7)
                pos_new = self.problem.generate_solution()
            else:
                p = np.tanh(np.abs(self.pop[idx].target.fitness - self.g_best.target.fitness))  # Eq.(2.2)
                vb = self.generator.uniform(-aa, aa, self.problem.n_dims)  # Eq.(2.3)
                vc = self.generator.uniform(-bb, bb, self.problem.n_dims)
                for jdx in range(0, self.problem.n_dims):
                    # two positions randomly selected from population
                    id_a, id_b = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                    if self.generator.uniform() < p:  # Eq.(2.1)
                        pos_new[jdx] = self.g_best.solution[jdx] + vb[jdx] * (self.weights[idx, jdx] * self.pop[id_a].solution[jdx] - self.pop[id_b].solution[jdx])
                    else:
                        pos_new[jdx] = vc[jdx] * pos_new[jdx]
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = agent
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)





img = imread('images/pacman.jpg') 
# if len(img.shape) > 2:
#     img = img[:, :, 0]  # Convert to grayscale if it's not already


gx, gy = np.gradient(img)
grad_magnitude = np.sqrt(gx**2 + gy**2)
grad_magnitude = grad_magnitude **2



@lru_cache(maxsize=1000) 
def cached_objective_function(solution_tuple):
    edge_map = np.array(solution_tuple).reshape(img.shape)
    matrix = np.sum((grad_magnitude - edge_map)**2)
    return matrix

def objective_function(solution):
    solution_tuple = tuple(solution)
    return cached_objective_function(solution_tuple)


problem_dict = {
    "bounds": BinaryVar(n_vars=img.size, name="edge_map"),
    "minmax": "min",
    "obj_func": objective_function
}


# The number of iterations that actually get run are epoch*pop_size
model = DevSMA(epoch=30000, pop_size=200, p_t = 0.05)

g_best = model.solve(problem_dict)
print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")


best_edge_map = np.array(g_best.solution).reshape(img.shape)




plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(best_edge_map, cmap='gray')
plt.title('Detected Edges')
plt.axis('off')

plt.show()


