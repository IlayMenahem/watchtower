from ramanujantools.cmf.known_cmfs import pFq
import sympy as sp
from sympy.abc import n
from ramanujantools import Position
from sympy import *
from implementations.simulatedAnnealing import simulated_annealing, simulated_diffusion
import pickle
import random
from functools import lru_cache

exact = False

if exact:
    constPrec = 100000
    p = 4
    q = 3
    x = sp.symbols(f'x:{p + 1}')
    y = sp.symbols(f'y:{q + 1}')
    x0, x1, x2, x3, x4 = x
    y0, y1, y2, y3= y

    Cal = symbols('Cal', real=True)
    Cal_value = Rational(1, 36) * (3 * polygamma(1, Rational(1, 3)) - polygamma(1, Rational(5, 6)))
    trueCalagari = Cal_value.evalf(constPrec)
else:
    with open('calagari.pkl', 'rb') as f:
        trueCalagari = pickle.load(f)

x = sp.symbols(f'x:{5}')
y = sp.symbols(f'y:{4}')
x0, x1, x2, x3, x4 = x
y0, y1, y2, y3 = y

ccmf = pFq(3, 2, sp.Rational(1,4))
init = Position({x0: sp.Rational(1,2), x1: sp.Rational(1,2), x2: sp.Rational(1,2), y0: sp.Rational(3,2), y1: sp.Rational(3,2)})


def normalized_depth(traj, depth):
    x = sp.symbols(f'x:{5}')
    y = sp.symbols(f'y:{4}')
    x0, x1, x2, x3, x4 = x
    y0, y1, y2, y3 = y

    avgStepSize = (abs(traj[x0])+abs(traj[x1])+abs(traj[x2])+abs(traj[y0])+abs(traj[y1]))/5

    return round(depth/avgStepSize)


def type1coloum(trajMat, depth):
  walk = trajMat.walk({n:1},depth,{n:1}).inv().T
  walk = walk/walk[0,0]

  return walk.col(0)


def delta(estemated, lim):
  error = sp.Abs(estemated -lim)
  denomenator = sp.denom(estemated)
  delta = -1-sp.log(error)/sp.log(denomenator)

  return (delta).evalf()


@lru_cache
def calagariDelta(trajectory, depth=5000):
    c1, c2 = sp.symbols('c1 ,c2')
    depth = normalized_depth(trajectory ,depth)

    trajMat = ccmf.trajectory_matrix(trajectory ,init)
    col1 = type1coloum(trajMat, depth)
    estematedCat = (8/(36*c1 + 36*c2 + 9)).subs({c1:col1[1] ,c2:col1[2]})
    deltaCurr = delta(estematedCat ,trueCalagari)

    return deltaCurr


def next_traj(traj):
    p = 3
    q = 2

    x = sp.symbols(f'x:{p + 1}')
    y = sp.symbols(f'y:{q + 1}')

    random_vec = [random.randint(-1, 1) for _ in range(5)]

    step = Position({x[0]: random_vec[0], x[1]: random_vec[1], x[2]: random_vec[2], y[0]: random_vec[3], y[1]: random_vec[4]})
    new_traj = traj + step

    return new_traj


if __name__ == '__main__':
    t0 = 1e8
    num_steps = 10000

    p = 4
    q = 3
    n = sp.symbols('n')
    x = sp.symbols(f'x:{p + 1}')
    y = sp.symbols(f'y:{q + 1}')
    x0, x1, x2, x3, x4 = x
    y0, y1, y2, y3 = y

    initialTraj = Position({x0: 5, x1: 6, x2: 6, y0: 15, y1: 18})

    best_state, best_cost, progress_best, progress_current, temperatures = simulated_annealing(
        t0, num_steps, initialTraj, next_traj, calagariDelta, minimize = False)
