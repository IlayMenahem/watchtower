from ramanujantools.cmf import CMF
from ramanujantools.cmf.known_cmfs import pi
from functools import lru_cache
from LIReC.db.access import db
import numpy as np
from sympy.abc import x, y
import random
from math import ceil
import matplotlib.pyplot as plt


def get_delta_func(cmf: CMF, start: dict, constant: str, distance: int, variables: tuple) -> callable:
    '''
    Returns a function that computes the delta for a given trajectory.

    Parameters:
    cmf (CMF): The CMF instance to use.
    start (dict): The starting point of the trajectory.
    constant (str): the constant we're interested in.
    distance (int): The distance to run the trajectory.
    variables (tuple): The variables involved in the trajectory.

    Returns:
    callable: A function that takes a trajectory and returns the delta.
    '''

    @lru_cache
    def get_delta(trajectory: tuple[int, ...]) -> float:
        gcd = np.gcd.reduce(trajectory)

        if gcd == 0:
            return -np.inf

        trajectory = tuple(int(val / gcd) for val in trajectory)
        trajectory_dict: dict = {var: val for var, val in zip(variables, trajectory)}

        iterations = ceil(distance / sum(abs(val) for val in trajectory))

        try:
            result_const = cmf.limit(trajectory_dict, iterations, start).as_float()
            delta = cmf.delta(trajectory_dict, iterations, start)
        except ZeroDivisionError:
            return -np.inf

        if not db.identify([result_const, constant]):
            return -np.inf

        return delta

    return get_delta


def random_step(trajectory: tuple[int, ...]):
    step = tuple(random.choice([-1, 0, 1]) for _ in trajectory)
    new_trajectory = tuple(a + b for a, b in zip(trajectory, step))

    return new_trajectory


def next_steps(trajectory: tuple[int, ...]):
    steps = []

    for i in range(len(trajectory)):
        step_up = list(trajectory)
        step_up[i] += 1
        steps.append(tuple(step_up))

        step_down = list(trajectory)
        step_down[i] -= 1
        steps.append(tuple(step_down))

    return steps


def get_neighbors_fn(cost_fn: callable):
    @lru_cache(maxsize=None, typed=True)
    def neighbors_fn(traj: tuple[int, ...]):
        neighbors = next_steps(traj)
        neighbors_cost = np.array([cost_fn(neighbor) for neighbor in neighbors])

        return neighbors, neighbors_cost

    return neighbors_fn


def quantize(x: float, y: float, res: int) -> tuple[int, int]:
    x_quantized = round(x * res)
    y_quantized = round(y * res)

    return x_quantized, y_quantized


def plot_search(progress_best, progress_current, temperatures):
    # plot the progress
    # Create two separate subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot costs on the first subplot
    ax1.plot(progress_best, label='Best Cost')
    ax1.plot(progress_current, label='Current Cost')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost')
    ax1.legend()
    ax1.set_title('Simulated Annealing Progress - Cost')

    # Plot temperature on the second subplot
    ax2.plot(temperatures, label='Temperature', color='red')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Temperature')
    ax2.legend()
    ax2.set_title('Simulated Annealing Progress - Temperature')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    cmf = pi()
    starting_point = {x: 1, y: 1}
    constant = 'pi'

    func = get_delta_func(cmf, starting_point, constant, 1500, (x, y))
    print(func((1, 1)))
