from functools import partial
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from termcolor import colored
from tqdm import tqdm

from ma_env import get_avai_actions, Trans, Reward
from mdp_agent import MDPAgent
from search_agent import dijkstra
from utils import Marker

np.random.seed(618)  # should not be put inside any function


class TreeNode(object):
    """
    Node class used in tree search:
    Parameters:
        type: 'MAX' (states) or 'EXP' (afterstates)
        height: >= 0
        locations: location tuple
        val: backpropagated long run return
        reward: immediate reward from the parent node and the branch action
    """

    def __init__(self, tp, h, location,
                 reward=None, prev_action=None):
        if tp not in ['MAX', 'EXP']:
            raise ValueError('No such node type!')
        self.id = np.random.rand()
        self.type = tp
        self.height = h
        self.location = location
        self.val = 0
        self.reward = reward
        self.children = []
        self.prev_action = prev_action

        # for mcts usage
        self.num_visit = 0
        self.child_indicies = []

        # for pUCT usage
        self.policy_prior = None

        # for recording actual action taken
        self.highlight = False

    def set_parent(self, p):
        self.parent = p


class MCTSAgent(MDPAgent):
    """
    MCTS with both chance nodes and decion nodes
    Implement heuristic (asymmetric) tree growth
    """

    def __init__(self, label, goal, randomness,
                 *,
                 gamma=0.99,
                 verbose=False,
                 reuse=False,
                 max_it=100,
                 explore_c=1,
                 pUCT=False,
                 pb_c=(1.25, 19652.0),
                 node_eval='SHORTEST_goal',
                 sample_select=10,
                 reward_scheme={'goal': 10, 'pitfall': -100, 'normal': -1}):
        super().__init__(label, goal, randomness, gamma, verbose)

        self.reuse = reuse  # reuse the previously built partial tree
        self.max_it = int(max_it)  # number of simulation
        self.c = explore_c  # for value guided mcts
        self.pUCT = pUCT  # for policy-value guided mcts
        self.pb_c_init, self.pb_c_base = pb_c

        # evaluation mode: 'IMMED', 'MDP', 'HEU', 'SHORTEST'
        self.node_eval = node_eval

        # for node selection
        self.sample_select = sample_select

        # example: {'goal': 0, 'pitfall': -1, 'normal': -1}
        self.reward_scheme = reward_scheme

    def act(self, state):
        N, prev_actions, locations, layout = state

        if locations[self.label] == self.goal:
            return 0

        # only happen in the first step
        if getattr(self, 'layout', None) is None:
            self.layout = layout
            self.pitfalls = list(map(tuple, np.array(np.where(layout == Marker.PITFALL)).T.tolist()))

            if self.node_eval.startswith('SHORTEST'):
                if self.node_eval.endswith('goal'):
                    self.Pi, self.dists = dijkstra(self.goal, self.layout)
                elif self.node_eval.endswith('pitfall'):
                    self.Pi, self.dists = dijkstra(self.pitfalls[0], self.layout, nogoods=[Marker.BLOCK])

                self.val_eval_fn = (
                    lambda loc:
                    (1 - self.gamma ** (self.dists[loc] - 1)) / (1 - self.gamma)
                    * self.reward_scheme['normal']
                    + self.gamma ** (self.dists[loc] - 1)
                    * self.reward_scheme['goal']
                )
                self.pi_eval_fn = (
                    lambda loc:
                    (np.eye(5)[self.Pi[str(loc)]] + 1e-1)
                    / (np.eye(5)[self.Pi[str(loc)]] + 1e-1).sum() if str(loc) in self.Pi
                    else np.eye(5)[0]
                )

                if self.verbose:
                    D = deepcopy(self.dists)
                    D[D == np.iinfo(np.int32).max] = -1
                    D[D <= 0] = - np.max(D)
                    print(D)
                    fig = plt.figure(figsize=(8, 5), dpi=150)
                    ax = plt.subplot(111, title=f'Shorest distance from each loc to goal')
                    c = ax.pcolor(D, cmap='Blues_r', edgecolors='k', linewidths=0.1)
                    ax.set_aspect('equal', adjustable='box')
                    ax.invert_yaxis()
                    plt.colorbar(c, ax=ax)

                    actions = ['stop', 'up', 'right', 'down', 'left']
                    for i in range(D.shape[0]):
                        for j in range(D.shape[1]):
                            if self.layout[i, j] == Marker.BLOCK:
                                continue
                            elif self.layout[i, j] == Marker.PITFALL:
                                ax.text(j + 0.5, i + 0.5,
                                        f'{int(D[i, j])}',
                                        fontsize=5, ha='center', va='center', color='black')
                            else:
                                ax.text(j + 0.5, i + 0.5,
                                        f'{int(D[i, j])}\n{actions[int(self.Pi[str((i, j))])]}',
                                        fontsize=5, ha='center', va='center', color='black')
                    fig.savefig('shortest.pdf', dpi=200)

            self.Trs = partial(
                Trans,
                goal=self.goal, nogoods=self.pitfalls, layout=self.layout, randomness=self.randomness
            )
            self.Rwd = partial(
                Reward,
                goal=self.goal, layout=self.layout, reward_scheme=self.reward_scheme
            )

        # Formulate a search tree based on the current state and prev_actions
        # Conduct tree search at every replanning
        self.policy = self.tree_search(locations[self.label], prev_actions)

        if self.verbose:
            print(f"Shortest dist to goal: {self.dists[locations[self.label]]}")

        avai_actions = get_avai_actions(locations[self.label], self.layout)
        unavai_actions = [a for a in range(5) if a not in avai_actions]
        child_num_visits = list(map(lambda c: c.num_visit, self.policy.children))
        child_num_visits = np.array(child_num_visits, dtype=float)
        action_values = list(map(lambda c: c.val / c.num_visit, self.policy.children))
        action_values = np.array(action_values, dtype=float)

        if self.pUCT:
            child_num_visits[unavai_actions] = -np.inf
            action = np.argmax(child_num_visits)
        else:
            action_values[unavai_actions] = -np.inf
            action = np.argmax(action_values)

        if self.verbose:
            print(colored(f"visit counts: {child_num_visits}", 'yellow'))
            print(colored(f"action values: {action_values}", 'yellow'))
            print(colored(f"eventually took {action}", 'yellow'))

        # if action not in get_avai_actions_mapf(locations[self.label], self.layout):
        #     action = 0

        # # 1. Naive strategy: iterative lessening
        # if getattr(self, 'prev_locations', None)\
        #         and locations == self.prev_locations\
        #         and action == 0:
        #     self.max_it = max(7, int(self.max_it * 0.6))

        # # 2. Naive strategy: iterative deepening
        # if getattr(self, 'prev_locations', None)\
        #         and locations == self.prev_locations\
        #         and action == 0:
        #     self.max_it = max(7, int(self.max_it * 1.2))

        self.prev_locations = locations
        return action

    def tree_search(self, curr_loc, prev_actions=None):
        if self.reuse and prev_actions is not None:
            # Assume:
            # 1. the taken action is a legal one -> if same loc, must have stopped
            # 2. all children of the root node are expanded
            action_taken = prev_actions[self.label]
            afterstate = self.root.children[action_taken]
            afterstate.highlight = True

            possible_succ_locs = tuple(map(lambda c: c.location, afterstate.children))
            if curr_loc in possible_succ_locs:
                idx = possible_succ_locs.index(curr_loc)
                self.root = self.root.children[action_taken].children[idx]
                self.root.parent = None
            else:
                new_root = TreeNode('MAX', self.root.height + 2, curr_loc, 0)  # set no parent
                self.root.children[action_taken].children.append(new_root)  # append no child_idx
                self.root = new_root
                print(colored('New root node!', 'red'))
        else:
            self.root = TreeNode('MAX', 0, curr_loc, 0)
            self.origin = self.root
        self.root.highlight = True

        max_height = self.root.height
        if self.verbose:
            iterator = tqdm(range(self.max_it))
        else:
            iterator = range(self.max_it)
        for it in iterator:
            node_to_exp = self.select(self.root, iteration=it)
            node_to_eval = self.expand(node_to_exp)
            max_height = max(max_height, node_to_eval.height)
            estimate_val = self.evaluate(node_to_eval)
            self.backup(node_to_eval, estimate_val)
        if self.verbose:
            print(f'Took {self.max_it} simus, lookahead for {(max_height - self.root.height) / 2} steps')
        return self.root

    def select(self, root, iteration):
        """
        Best-first node selection:
        V_s / N_s + sqrt(2N / N_s)
        Note that Q(s, a) = V_s / N_s is not stationary
        Upon each chosen action, sample a succ_state by the transition model
        """
        curr_node = root  # the iterating node is always a MAX node
        selected_a = 0
        while True:
            # a state whose action has not been fully expanded or is a leaf node
            if len(curr_node.children) < 5:
                return curr_node

            # or, iteratively find the best action with UCB heuristic
            if self.pUCT:
                Qs = _puct(curr_node)
            else:
                Qs = _vanilla_uct(curr_node, iteration)

            avai_actions = get_avai_actions(curr_node.location, self.layout)
            unavai_actions = [a for a in range(5) if a not in avai_actions]
            Qs[unavai_actions] = -np.inf
            selected_a = np.argmax(Qs)

            successor_idx = self.sample_from_successors(curr_node, selected_a)
            curr_node = curr_node.children[selected_a].children[successor_idx]

    def expand(self, node_to_exp):
        """
        Given a MAX node, try a not-chosen action, and a new EXP node
        Then sample a new child MAX node with belief revision
        """
        # correct?: check if node_to_exp is new
        if node_to_exp.num_visit == 0:
            return node_to_exp

        expand_a = len(node_to_exp.children)
        succ_node = TreeNode('EXP', node_to_exp.height + 1,
                             location=node_to_exp.location,
                             prev_action=expand_a)
        node_to_exp.children.append(succ_node)
        succ_node.set_parent(node_to_exp)
        successor_idx = self.sample_from_successors(node_to_exp, expand_a)

        return node_to_exp.children[expand_a].children[successor_idx]

    def sample_from_successors(self, node, action):
        """
        Given a MAX node and an action,
        sample an action profile for the other agents,
        return the index of the sampled profile
        """
        successors, probs = self.Trs(node.location, action)

        successor_idx = np.random.choice(range(5), p=probs)
        if successor_idx in node.children[action].child_indicies:
            return node.children[action].child_indicies.index(successor_idx)
        else:
            succ_loc = successors[successor_idx]
            rwd = self.Rwd(node.location, action, succ_loc)
            succ_node = TreeNode('MAX', node.height + 2,  # exp+1 and max+1
                                 location=succ_loc,
                                 reward=rwd,
                                 prev_action=action)
            node.children[action].children.append(succ_node)
            node.children[action].child_indicies.append(successor_idx)
            succ_node.set_parent(node.children[action])
            return -1  # i.e., that lastest one

    def evaluate(self, node_to_eval):
        """
        By enquiring the node evaluation function
        """
        if self.node_eval.startswith('SHORTEST'):
            val = self.val_eval_fn(node_to_eval.location)
            node_to_eval.policy_prior = self.pi_eval_fn(node_to_eval.location)
            return val
        elif self.node_eval == 'MC':
            raise NotImplementedError
        else:
            return 0

    def backup(self, evaled_node, estimate_val):
        future_val = estimate_val
        curr_node = evaled_node

        curr_node.val += future_val
        curr_node.num_visit += 1
        while getattr(curr_node, 'parent', None) is not None:
            # MAX -> EXP
            future_val = self.gamma * future_val + curr_node.reward
            curr_node = curr_node.parent
            curr_node.val += future_val
            curr_node.num_visit += 1

            # EXP -> MAX
            curr_node = curr_node.parent
            curr_node.val += future_val
            curr_node.num_visit += 1

    def close(self):
        """
        If reuse=True, this plots the whole planning tree,
        else, this only plots the tree built by the final step
        """
        if self.verbose > 1:
            vis_tree(self.origin)


def _vanilla_uct(node, iter,
                 *,
                 c=1):
    node_visit = node.num_visit
    child_visit = np.array(list(map(lambda n: n.num_visit, node.children)))
    Qs = _qtransform(node)
    ucb = c * np.sqrt(2 * np.log(node_visit) / child_visit)
    tie_breaking_noise = np.random.uniform(size=len(child_visit)) * 1e-10
    return Qs + ucb + tie_breaking_noise


def _puct(node,
          *,
          pb_c_init=0.5,  # deepmind: 1.25,
          pb_c_base=19652.0,
          seed=0):
    """
    The pUCT formula from muZero:
    given a `parent` node, return the pUCT score for its children.
    `1 + child_visit` because it is initialzed to 0
    """
    # TODO
    # 1. num_visit init to 0 or 1?
    # 2. how q be normalized to [0, 1]?
    #    2.1 should be pass a parent node?
    node_visit = node.num_visit
    child_visit = np.array(list(map(lambda n: n.num_visit, node.children)))
    # TODO: if a leaf node is reused, num_visit will equal to child_visit + 1
    # if node_visit != sum(child_visit):
    #     raise ValueError(f"{node_visit} not equal to sum of {child_visit}")
    prior_probs = (node.policy_prior + 1e-1) / np.sum(node.policy_prior + 1e-1)
    pb_c = pb_c_init + np.log((node_visit + pb_c_base + 1) / pb_c_base)
    policy_prior = prior_probs * np.sqrt(node_visit) * pb_c / (child_visit)
    Qs = _qtransform(node)
    tie_breaking_noise = np.random.uniform(size=len(child_visit)) * 1e-10
    return Qs + policy_prior + tie_breaking_noise


def _qtransform(node, *, epsilon=1e-8):
    """
    Given a parent node,
    return Qs for its children normalized to [0, 1].
    Normalization is done w.r.t. the parent and siblings.
    """
    child_Qs = np.array(list(map(lambda x: x.val / x.num_visit, node.children)))
    node_Q = node.val / node.num_visit
    lo = np.minimum(node_Q, np.min(child_Qs))
    hi = np.maximum(node_Q, np.max(child_Qs))
    return (child_Qs - lo) / np.maximum((hi - lo), epsilon)


def vis_tree(root):
    import pygraphviz as pgv

    def node2str(node):
        if node.type == 'MAX':
            return (f"Loc: {node.location}\n"
                    f"Reward: {node.reward}\n"
                    f"EstimateVal: {(node.val / node.num_visit):.2f}\n"
                    f"Visits: {node.num_visit}\n"
                    f"Pi priors: {node.policy_prior.round(2)}")
        else:
            a_list = ['stop', 'up', 'right', 'down', 'left']
            return (f"Action: {a_list[node.prev_action]}\n"
                    f"EstimateVal: {(node.val / node.num_visit):.2f}")

    def add_nodes_and_edges(graph, node):
        # Add node with different colors based on type
        if node.type == "MAX":
            graph.add_node(node.id, label=node2str(node), color='red', shape='ellipse')
        else:
            graph.add_node(node.id, label=node2str(node), color='lightgreen', shape='box')

        for child in node.children:
            if node.highlight and child.highlight:
                graph.add_edge(node.id, child.id, color='coral1', penwidth=5)
            else:
                graph.add_edge(node.id, child.id)
            add_nodes_and_edges(graph, child)

    G = pgv.AGraph(directed=True)
    add_nodes_and_edges(G, root)

    # other Graphviz layout engines: 'neato', 'fdp', etc.
    G.draw('tree_visualization.pdf',
           prog='dot', format='pdf', args='-Gdpi=300 -s600,360')
