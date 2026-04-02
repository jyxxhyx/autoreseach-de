"""
Autoresearch DE optimization script. Single-run, single-file.
Agent may modify ANYTHING in this file: strategy, operators, hyperparameters, etc.

Usage:
    uv run train.py
"""

import time
import numpy as np
from prepare import load_problem, evaluate_solution, DIM, BOUNDS, TIME_BUDGET

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly — this is the agent's playground)
# ---------------------------------------------------------------------------
POP_SIZE = 500          # 种群规模
F = 0.8                 # 缩放因子
CR = 0.9                # 交叉概率
STRATEGY = "rand1"      # 'rand1' or 'best1'
MAX_GENERATIONS = 5000  # 硬性上限（实际由 TIME_BUDGET 提前截断）

# ---------------------------------------------------------------------------
# Differential Evolution (modifiable algorithm)
# ---------------------------------------------------------------------------

class DifferentialEvolution:
    def __init__(self, obj_func, dim, bounds, pop_size, F, CR, max_gen, strategy):
        self.obj_func = obj_func
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.max_gen = max_gen
        self.strategy = strategy
        self.best_solution = None
        self.best_fitness = float("inf")
        self.convergence_curve = []

    def _init_population(self):
        lower, upper = self.bounds
        return np.random.uniform(lower, upper, (self.pop_size, self.dim))

    def _boundary_handling(self, mutant):
        lower, upper = self.bounds
        for i in range(self.dim):
            if mutant[i] < lower or mutant[i] > upper:
                mutant[i] = np.random.uniform(lower, upper)
        return mutant

    def run(self, time_budget):
        pop = self._init_population()
        fitness = np.array([self.obj_func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        self.best_fitness = fitness[best_idx]
        self.best_solution = pop[best_idx].copy()
        self.convergence_curve.append(self.best_fitness)

        t_start = time.time()
        total_eval_time = 0.0

        for gen in range(1, self.max_gen + 1):
            t_gen_start = time.time()

            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                r1, r2, r3 = np.random.choice(indices, 3, replace=False)

                if self.strategy == "rand1":
                    mutant = pop[r1] + self.F * (pop[r2] - pop[r3])
                elif self.strategy == "best1":
                    mutant = self.best_solution + self.F * (pop[r1] - pop[r2])
                else:
                    raise ValueError(f"Unknown strategy: {self.strategy}")

                mutant = self._boundary_handling(mutant)

                trial = np.copy(pop[i])
                j_rand = np.random.randint(0, self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]

                trial_fitness = self.obj_func(trial)
                if trial_fitness <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial.copy()

            self.convergence_curve.append(self.best_fitness)
            total_eval_time += time.time() - t_gen_start

            if gen % 100 == 0:
                print(f"\rgen {gen:05d} | best_fitness: {self.best_fitness:.6e} "
                      f"| gen_time: {time.time()-t_gen_start:.3f}s | "
                      f"budget_used: {total_eval_time:.1f}/{time_budget}s", end="", flush=True)

            if total_eval_time >= time_budget:
                print(f"\nTime budget exhausted at generation {gen}.")
                break

        print()
        return self.best_solution, self.best_fitness, total_eval_time


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(None)  # 算法内部的随机性每次不同，但面对的问题地形相同
    t_total_start = time.time()

    problem = load_problem()

    de = DifferentialEvolution(
        obj_func=lambda x: evaluate_solution(x, problem),
        dim=DIM,
        bounds=BOUNDS,
        pop_size=POP_SIZE,
        F=F,
        CR=CR,
        max_gen=MAX_GENERATIONS,
        strategy=STRATEGY,
    )

    best_x, best_fitness, training_seconds = de.run(TIME_BUDGET)

    t_total_end = time.time()

    # --- 标准化输出（Agent 通过 grep 提取这些行）---
    print("---")
    print(f"best_fitness: {best_fitness:.6e}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds: {t_total_end - t_total_start:.1f}")
    print(f"pop_size: {POP_SIZE}")
    print(f"F: {F}")
    print(f"CR: {CR}")
    print(f"strategy: {STRATEGY}")
    print(f"total_generations: {len(de.convergence_curve)}")
