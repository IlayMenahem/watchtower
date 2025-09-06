from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import csv
from ramanujantools.cmf.known_cmfs import pi
import sympy as sp
from utils import DeltaCalculator
from scipy import stats
import numpy as np


def generate_trajs(num_trajectories, redius):
    trajectories = []

    def remove_equivalents(trajectories):
        unique_trajectories = ()

        for traj in trajectories:
            gcd = np.gcd.reduce(traj)

            if gcd == 0:
                continue

            traj = tuple(int(val / gcd) for val in traj)

            if traj not in unique_trajectories:
                unique_trajectories += (traj,)

        return unique_trajectories


    while len(trajectories) < num_trajectories:
        traj = tuple(stats.randint.rvs(0, redius, size=2))

        if redius >= np.linalg.norm(traj) and traj not in trajectories:
            trajectories.append(traj)

    trajectories = remove_equivalents(trajectories)

    return trajectories

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

    cost_fn = DeltaCalculator(cmf, starting_point, constant, 5000, variables)

    # sample trajectories
    num_trajectories = 1500
    redius = 100
    trajectories = generate_trajs(num_trajectories, redius)

    print(f"Generated {len(trajectories)} unique trajectories.")

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()-2) as executor:
        deltas = list(executor.map(cost_fn, trajectories))

    # save to a csv file the states and deltas
    with open('states_deltas.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'delta'])
        for i, (x, y) in enumerate(trajectories):
            writer.writerow([x, y, deltas[i]])
