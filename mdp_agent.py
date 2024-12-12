import os
import pickle
import time
from collections import namedtuple
from functools import partial
from itertools import product
from copy import deepcopy

import numpy as np
from mdptoolbox import mdp
from tqdm import tqdm

from ma_env import Trans, Reward
from utils import hash, Marker


class MDPAgent(object):
    """
    MDP agent
    """

    def __init__(self, label, goal, randomness, gamma=0.99, verbose=False):
        super().__init__()
        self.label = label
        self.goal = goal
        self.randomness = randomness
        self.gamma = gamma

        self.verbose = verbose

    def act(self, state):
        N, prev_actions, locations, layout = state

        if locations[self.label] == self.goal:
            return 0

        if getattr(self, 'layout', None) is None:
            self.layout = layout
            self.pitfalls = list(map(tuple, np.array(np.where(layout == Marker.PITFALL)).T.tolist()))
            self.policy = self.translate_solve()

        Si = self.S.index(locations[self.label])
        action = self.policy[Si]

        return action

    def translate_solve(self):
        """
        Formulate an MDP from the pivotal agent's perspective,
        and invoke mdptoolbox
        """
        t0 = time.time()
        if getattr(self, 'layout', None) is None:
            raise RuntimeError("Get an invalid layout!")

        # Get all possible states
        if getattr(self, 'S', None) is None:
            self.S = []
            for m in Marker.ACCESSIBLE:
                self.S += list(map(tuple, np.array(np.where(self.layout == m)).T))
            self.num_all = len(self.S)

        S = self.S
        num_all = self.num_all

        Trs = partial(
            Trans,
            goal=self.goal, nogoods=self.pitfalls, layout=self.layout, randomness=self.randomness
        )
        Rwd = partial(
            Reward,
            goal=self.goal, layout=self.layout
        )

        # Translate transition/reward matrix, shape(T) := (A,S,S), shape(R) := (A,S,S)
        T = np.zeros(shape=(5, num_all, num_all))
        R = np.zeros(shape=(5, num_all, num_all))
        for i, Si in enumerate(S):
            for A in range(5):
                successors, probs = Trs(Si, A)  # [(1, 6), ..., (5, 2)], [0.1, 0.1, 0.1, 0.6, 0.1]
                for a, Sj in enumerate(successors):
                    j = S.index(Sj)
                    T[A, i, j] += probs[a]
                    R[A, i, j] = Rwd(Si, A, Sj)

        # T is conceptually a stochastic matrix already,
        # But due to float ops, we need to further normalize it
        T = T / np.sum(T, axis=2).reshape(5, num_all, 1)

        t1 = time.time()
        VI = mdp.ValueIteration(T, R, discount=self.gamma)
        VI.run()
        t2 = time.time()

        if self.verbose:
            from matplotlib import pyplot as plt
            print(f"Translation costs {t1 - t0}s")
            print(f"VI costs {t2 - t1}s")
            print(f"Did {VI.iter} iterations")

            Pi = np.full_like(self.layout, fill_value=-10)
            Vmap = np.full_like(self.layout, fill_value=1.2 * min(VI.V))
            for i, Si in enumerate(S):
                Vmap[Si] = VI.V[i]
                Pi[Si] = VI.policy[i]
            print(Vmap)
            print(Pi)

            plt.figure(figsize=(8, 5), dpi=150)
            ax = plt.subplot(111, title=f'Optimal value wrt p={self.randomness}')
            c = ax.pcolor(Vmap, cmap='Blues_r', edgecolors='k', linewidths=0.1)
            ax.set_aspect('equal', adjustable='box')
            ax.invert_yaxis()
            plt.colorbar(c, ax=ax)

            actions = ['stop', 'up', 'right', 'down', 'left']
            for i in range(Vmap.shape[0]):
                for j in range(Vmap.shape[1]):
                    if self.layout[i, j] == Marker.BLOCK:
                        continue
                    ax.text(j + 0.5, i + 0.5,
                            f'{Vmap[i, j]:.2f}\n{actions[int(Pi[i, j])]}',
                            fontsize=5, ha='center', va='center', color='black')
            plt.savefig('vi.pdf', dpi=200)
        return VI.policy

    def close(self):
        pass
