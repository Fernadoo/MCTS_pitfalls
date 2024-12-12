import random
from collections import namedtuple
from copy import deepcopy
import numpy as np

from utils import move, Marker


State = namedtuple('State',
                   ['NumAgents', 'PrevActions', 'Locations', 'Layout'])


class MAPF(object):
    """docstring for MAPF"""

    def __init__(self, agents, starts, goals, layout, randomness=0.3, stop_on=None):
        """
        agents: a list of agents
        starts/goals: a list of tuples
        layout: a 2d array; 0-empty, 1-obstacle
        """
        super(MAPF, self).__init__()
        self.agents = agents
        self.N = len(agents)
        self.starts = tuple(starts)
        self.goals = tuple(goals)
        self.layout = layout

        self.randomness = randomness

        self.state = State(self.N, None, self.starts, self.layout)
        self.reach_goal = np.zeros(self.N)
        self.max_steps = max(self.layout.shape) * 3
        self.penalty_steps = max(self.layout.shape) * 10
        self.stop_on = stop_on
        self.num_collisions = np.zeros(self.N)

    def transit(self, action_profile):
        """
        Given a state and an action profile,
        returns a successor state or a vector of prob.
        """
        N, prev_actions, locations, layout = self.state
        avai_actions = self.get_avai_actions()

        succ_locations = []
        for i in range(N):
            if action_profile[i] not in avai_actions[i]:
                # raise RuntimeError(f"Action {action_profile[i]} not allowed in {locations}!")
                action_profile[i] = 0
            # if random.random() < self.randomness:
            #     succ_loc = move(locations[i], random.choice(avai_actions[i]))
            # else:
            #     succ_loc = move(locations[i], action_profile[i])
            if random.random() < self.randomness:
                action_profile[i] = random.choice(avai_actions[i])
            succ_loc = move(locations[i], action_profile[i])
            succ_locations.append(succ_loc)

        self.state = State(N, action_profile, tuple(succ_locations), layout)
        return self.state

    def get_avai_actions(self):
        """
        Given a state,
        returns feasible actions for each agent.
        Only forbids actions into walls or beyond the map
        """
        N, prev_actions, locations, layout = self.state
        nrows = len(self.layout)
        ncols = len(self.layout[0])

        avai_actions = []
        for i in range(N):
            loc = locations[i]
            actions = []
            for a in range(5):
                succ_loc = move(loc, a)
                if succ_loc[0] not in range(nrows) or succ_loc[1] not in range(ncols) or\
                        layout[succ_loc] in Marker.INACCESSIBLE:
                    continue
                actions.append(a)
            avai_actions.append(actions)

        return avai_actions

    def check_end(self):
        """
        Given a state,
        returns whether it is an end.
        """
        _, _, locations, _ = self.state
        for i, loc in enumerate(locations):
            if loc != self.goals[i]:
                self.reach_goal[i] = min(self.reach_goal[i] + 1, self.penalty_steps)
        if self.stop_on is not None:
            if locations[self.stop_on] == self.goals[self.stop_on]:
                return True
        else:
            if locations == self.goals:
                return True
        return False

    def run(self):
        """
        Returns history: a list of aux info at each state,
        In this env, returns a list of (actions, succ_locations)
        """
        history = [(None, self.state.Locations)]
        self.step = 0
        stuck_indicator = 0
        stuck_for_n_steps = 5
        STUCK = False
        while not self.check_end():
            action_profile = []
            for i in range(self.N):
                action_profile.append(self.agents[i].act(self.state))
            _, action_profile, locations, _ = self.transit(action_profile)
            collisions = check_collision(history[-1][-1], locations)
            for i, c_i in enumerate(collisions):
                if c_i:
                    # self.reach_goal[i] = self.penalty_steps
                    self.num_collisions[i] += 1
            history.append((deepcopy(action_profile), locations))
            self.step += 1
            if self.step > self.max_steps:
                break

            # check if stuck forever
            if locations == history[-2][1]:
                stuck_indicator += 1
                if stuck_indicator >= stuck_for_n_steps:
                    STUCK = True
                    break
            else:
                stuck_indicator = 0

        for i in range(self.N):
            self.agents[i].close()
        return history, self.reach_goal, self.num_collisions, STUCK


def check_collision(prev_locations, curr_locations):
    N = len(prev_locations)
    collisions = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            collide = False
            if (prev_locations[i], prev_locations[j]) == (curr_locations[j], curr_locations[i]):
                collide = True
            if curr_locations[i] == curr_locations[j]:
                collide = True
            if collide:
                collisions[i].append(j)
                collisions[j].append(i)
    return collisions


def get_avai_actions(Si, layout):
    nrows, ncols = layout.shape
    avai_actions = []
    for a in range(5):
        succ_loc = move(Si, a)
        if succ_loc[0] not in range(nrows) or succ_loc[1] not in range(ncols) or\
                layout[succ_loc] in Marker.INACCESSIBLE:
            continue
        avai_actions.append(a)
    return avai_actions


def Trans(Si, A, goal, nogoods, layout, randomness):
    """
    Given Si and action, returns a distribution over successor Sj's
    `goal`, `layout` and `randomness` should first be grounded when the agent is initialized
    """
    nrows, ncols = layout.shape

    probs = np.zeros(5) + randomness / 5
    probs[A] += (1 - randomness)

    if Si == goal:
        return [Si] * 5, probs
    elif Si in nogoods:
        return [Si] * 5, probs

    successors = []
    for a in range(5):
        succ_loc = move(Si, a)
        if succ_loc[0] not in range(nrows) or succ_loc[1] not in range(ncols) or\
                layout[succ_loc] in Marker.INACCESSIBLE:
            successors.append(Si)
        else:
            successors.append(succ_loc)
    return successors, probs


def Reward(Si, A, Sj, goal, layout, *, reward_scheme={'goal': 0, 'pitfall': -1, 'normal': -1}):
    """
    Given (Si, A, Sj), returns the reward according to the reward scheme
    `goal` and `reward_scheme` should first be grounded when the agent is initialized
    A legal reward_scheme example: {'goal': 10, 'pitfall': -20, 'normal':-1}
    """
    if Sj == goal:
        return reward_scheme['goal']
    elif layout[Sj] == Marker.PITFALL:
        return reward_scheme['pitfall']
    else:
        return reward_scheme['normal']
