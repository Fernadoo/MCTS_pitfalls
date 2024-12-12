from queue import PriorityQueue
from collections import namedtuple

import numpy as np

from utils import move, rev_action, hash, euc_dist, Marker

"""
A-STAR SEARCH
Inputs: init, goal, layout
Returns: a sequence (list) of actions
"""


def astar(init, goal, layout, nogoods=[Marker.BLOCK, Marker.PITFALL]):
    """
    Ignore the others, simply do astar search,
    and always stick to the plan, never replan
    """
    Node = namedtuple('ANode',
                      ['fValue', 'gValue', 'PrevAction', 'Loc'])
    nrows = len(layout)
    ncols = len(layout[0])

    def get_successors(node):
        f, g, prev_action, curr_loc = node
        successors = []
        for a in range(5):
            succ_loc = move(curr_loc, a)
            if succ_loc[0] not in range(nrows) or succ_loc[1] not in range(ncols) or\
                    layout[succ_loc] in nogoods:
                continue
            heu = euc_dist(succ_loc, goal)
            succ_node = Node(heu + g + 1, g + 1, a, succ_loc)
            successors.append(succ_node)
        return successors

    plan = []
    visited = []
    parent_dict = dict()
    q = PriorityQueue()
    q.put(Node(euc_dist(init, goal), 0, None, init))
    while not q.empty():
        curr_node = q.get()
        if curr_node.Loc == goal:
            # backtrack to get the plan
            curr = curr_node
            while curr.Loc != init:
                plan.insert(0, curr.PrevAction)
                curr = parent_dict[curr]
            return plan

        if curr_node.Loc in visited:
            continue
        successors = get_successors(curr_node)
        for succ_node in successors:
            q.put(succ_node)
            parent_dict[succ_node] = curr_node
        visited.append(curr_node.Loc)
    raise RuntimeError("No astar plan found!")


class AStarAgent(object):
    """docstring for AStarAgent"""

    def __init__(self, label, goal):
        """
        label: an integer name
        """
        super(AStarAgent, self).__init__()
        self.label = label
        self.goal = goal
        self.plan = None
        self.round = 0

    def act(self, state):
        N, prev_actions, locations, layout = state

        if self.plan is None:
            self.plan = astar(locations[self.label], self.goal, layout)
        if locations[self.label] == self.goal:
            return 0

        if self.round < len(self.plan):
            action = self.plan[self.round]
            self.round += 1
        else:
            action = 0
        return action

    def close(self):
        pass


class ReplanAStarAgent(AStarAgent):
    """docstring for SafeAgent"""

    def act(self, state):
        N, prev_actions, locations, layout = state
        self.plan = astar(locations[self.label], self.goal, layout)
        return self.plan[0]


"""
DIJKSTRA SEARCH
Inputs: init, layout
Outputs: a policy (loc -> action) = shortest path tree from every goal to init
"""


def dijkstra(init, layout, nogoods=[Marker.BLOCK, Marker.PITFALL]):
    """
    Dijkstra for directly computing single source (goal) policy, and distances
    """
    Node = namedtuple('DNode',
                      ['gValue', 'PrevAction', 'Loc'])
    nrows, ncols = layout.shape
    num_valid = sum(list(map(lambda x: x not in nogoods, layout.reshape(-1))))

    def get_successors(node):
        g, prev_action, curr_loc = node
        successors = []
        for a in range(5):
            succ_loc = move(curr_loc, a)
            if succ_loc[0] not in range(nrows) or succ_loc[1] not in range(ncols) or\
                    layout[succ_loc] in nogoods:
                continue
            succ_node = Node(g + 1, a, succ_loc)
            successors.append(succ_node)
        return successors

    visited = []
    policy = dict()
    dists = np.full_like(layout, fill_value=np.iinfo(np.int32).max)
    curr_num_visited = 0
    q = PriorityQueue()
    q.put(Node(0, 0, init))
    while not q.empty() and curr_num_visited < num_valid:
        curr_node = q.get()
        if curr_node.Loc in visited:
            continue
        curr_num_visited += 1
        successors = get_successors(curr_node)
        for succ_node in successors:
            q.put(succ_node)
        visited.append(curr_node.Loc)
        policy[hash(curr_node.Loc)] = rev_action(curr_node.PrevAction)
        dists[curr_node.Loc] = curr_node.gValue
    print(curr_num_visited, num_valid)
    if curr_num_visited < num_valid:
        raise RuntimeError("No dijkstra plan found!")
    return policy, dists


class DijkstraAgent(object):
    """docstring for DijkstraAgent"""

    def __init__(self, label, goal):
        super(DijkstraAgent, self).__init__()
        self.label = label
        self.goal = goal
        self.policy = None

    def act(self, state):
        N, prev_actions, locations, layout = state

        if self.policy is None:
            self.policy, dists = dijkstra(self.goal, layout)
            print(dists)
            # for loc in self.policy:
            #     print(loc, self.policy[loc])
        if locations[self.label] == self.goal:
            return 0

        action = self.policy[hash(locations[self.label])]
        return action

    def close(self):
        pass
