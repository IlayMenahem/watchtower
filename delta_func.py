from ramanujantools.cmf import CMF
from ramanujantools.cmf.known_cmfs import pi
from functools import lru_cache
import numpy as np
from sympy.abc import x, y


def get_delta_func(cmf: CMF, start: dict, constant: str, iterations: int) -> callable:
    '''
    Returns a function that computes the delta for a given trajectory.

    Parameters:
    cmf (CMF): The CMF instance to use.
    start (dict): The starting point of the trajectory.
    constant (str): the constant we're interested in.

    Returns:
    callable: A function that takes a trajectory and returns the delta.
    '''

    @lru_cache(maxsize=None, typed=True)
    def get_delta(trajectory: dict):
        result_const = cmf.limit(trajectory, iterations, start).as_float()
        delta = cmf.delta(trajectory, iterations, start)

        if not db.identify([result_const, constant]):
            return -np.inf

        return delta

    return get_delta


if __name__ == '__main__':
    cmf = pi()
    starting_point = {x: 1, y: 1}
    constant = 'pi'

    func = get_delta_func(cmf, starting_point, constant, 1500)
    print(func({x: 1, y: 1}))
