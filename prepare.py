"""
One-time problem preparation for autoresearch DE experiments.
Generates the Shifted Rotated Rosenbrock problem instance and provides the fixed evaluation interface.

Usage:
    python prepare.py
"""

import os
import pickle
import numpy as np

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------
DIM = 300                  # 问题维度
BOUNDS = [-100, 100]      # 搜索空间边界
TIME_BUDGET = 300         # 优化时间预算，单位：秒（5分钟）
EVAL_SEED = 42            # 问题实例化的随机种子（锁定问题本身）
np.random.seed(EVAL_SEED)

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "de_autoresearch")
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Problem definition (fixed landscape)
# ---------------------------------------------------------------------------

def _generate_rotation_matrix(dim):
    """使用QR分解生成随机正交矩阵"""
    H = np.random.randn(dim, dim)
    Q, R = np.linalg.qr(H)
    return Q


class ShiftedRotatedRosenbrock:
    def __init__(self, dim):
        self.dim = dim
        self.bounds = BOUNDS
        self.shift = np.random.uniform(-80, 80, dim)
        self.rotation_matrix = _generate_rotation_matrix(dim)
        self.global_optimum_fitness = 0.0

    def evaluate(self, x):
        z = x - self.shift
        y = np.dot(self.rotation_matrix, z)
        fitness = 0.0
        for i in range(self.dim - 1):
            term1 = (y[i] - 1) ** 2
            term2 = 100 * (y[i+1] - y[i]**2) ** 2
            fitness += term1 + term2
        return fitness


# ---------------------------------------------------------------------------
# One-time generation & serialization
# ---------------------------------------------------------------------------

def generate_problem():
    problem_path = os.path.join(CACHE_DIR, "problem.pkl")
    if os.path.exists(problem_path):
        print(f"Problem: already generated at {problem_path}")
        return
    problem = ShiftedRotatedRosenbrock(DIM)
    with open(problem_path, "wb") as f:
        pickle.dump(problem, f)
    print(f"Problem: generated and saved to {problem_path}")
    print(f"  dim={DIM}, bounds={BOUNDS}, shift_norm={np.linalg.norm(problem.shift):.2f}")


def load_problem():
    problem_path = os.path.join(CACHE_DIR, "problem.pkl")
    assert os.path.exists(problem_path), "Problem not found. Run prepare.py first."
    with open(problem_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Fixed evaluation interface (DO NOT CHANGE — this is the ground truth metric)
# ---------------------------------------------------------------------------

def evaluate_solution(x, problem):
    """Evaluate a single solution. Returns fitness (lower is better)."""
    return problem.evaluate(x)


if __name__ == "__main__":
    generate_problem()
