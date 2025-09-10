from implementations.simulatedAnnealing import simulated_annealing
from ramanujantools.cmf.known_cmfs import pi
import sympy as sp
from watchtower.utils import random_step, get_delta_func, plot_search


if __name__ == '__main__':
    t0 = 100
    num_steps = 1000

    p = 1
    q = 1

    cmf = pi()
    constant = 'pi'

    x, y = sp.symbols('x, y')
    variables = (x, y)
    starting_point = {x: 1, y: 1}

    cost_fn = get_delta_func(cmf, starting_point, constant, 5000, variables)

    state0 = (1, 1)
    best_state, best_cost, progress_best, progress_current, temperatures = simulated_annealing(t0, num_steps, state0, random_step, cost_fn, minimize=False)

    print(f"Best state: {best_state}")
    print(f"Best cost: {best_cost}")

    plot_search(progress_best, progress_current, temperatures)
