import argparse
import random

import numpy as np

from search_agent import AStarAgent, ReplanAStarAgent, DijkstraAgent
from mdp_agent import MDPAgent
from mcts_agent import MCTSAgent
from ma_env import MAPF
from animator import Animation
from utils import parse_map_from_file, parse_locs, show_args, Marker


def get_args():
    parser = argparse.ArgumentParser(
        description='Pitfalls for Single-Agent Planning.'
    )

    parser.add_argument('--agents', dest='agents', type=int, default=0,
                        help='Specify the number of agents')
    parser.add_argument('--map', dest='map', type=str,
                        help='Specify a map')
    parser.add_argument('--starts', dest='starts', type=str, nargs='+',
                        help='Specify the starts for each agent, '
                             'e.g., 2_0 0_2, or simple type `random`')
    parser.add_argument('--goals', dest='goals', type=str, nargs='+',
                        help='Specify the goals for each agent, '
                             'e.g., 5_1 6_4, or simple type `random`')
    parser.add_argument('--randomness', dest='randomness', type=float, default=0.3,
                        help='Specify the extent of randomness of the env')
    parser.add_argument('--it', dest='it', type=int, default=100,
                        help='Specify the number of iterations for MCTS')
    parser.add_argument('--verbose', dest='verbose', type=int, default=0,
                        help='Specify the need of showing debug info')
    parser.add_argument('--vis', dest='vis', action='store_true',
                        help='Visulize the process')
    parser.add_argument('--save', dest='save', type=str,
                        help='Specify the path to save the animation')

    args = parser.parse_args()
    args.map = parse_map_from_file(args.map)
    empty_cells = np.array(np.where(args.map == Marker.CELL)).T.tolist()
    print(f"Number of empty cells: {len(empty_cells)}")
    if 'random' in args.starts:
        candidate = random.sample(empty_cells, k=args.agents)
        args.starts = list(map(tuple, candidate))
    else:
        args.starts = parse_locs(args.starts)
    if 'random' in args.goals:
        candidate = random.sample(empty_cells, k=args.agents)
        args.goals = list(map(tuple, candidate))
    else:
        args.goals = parse_locs(args.goals)
    if args.save:
        args.save = 'results/' + args.save

    return args


def show_hist(history):
    action_list = ['stop', 'up', 'right', 'down', 'left']
    for t, info in enumerate(history):
        actions, locations = info
        if t == 0:
            print(f'T{t}: start from {locations}')
        else:

            print(f'T{t}: '
                  f'actions: {list(map(lambda a: action_list[a], actions))}\t'
                  f'locations: {locations}')


if __name__ == '__main__':

    args = get_args()
    show_args(args)

    agents = []
    # agents.append(AStarAgent(0, args.goals[0]))
    # agents.append(ReplanAStarAgent(0, args.goals[0]))
    # agents.append(DijkstraAgent(0, args.goals[0]))
    # agents.append(MDPAgent(0, args.goals[0], randomness=args.randomness, verbose=args.verbose))
    agents.append(MCTSAgent(0, args.goals[0],
                            randomness=args.randomness, reuse=False, max_it=args.it,
                            pUCT=False, node_eval='SHORTEST_goal',
                            verbose=args.verbose))
    # agents.append(MCTSAgent(0, args.goals[0],
    #                         randomness=args.randomness, reuse=False, max_it=args.it,
    #                         pUCT=False, node_eval='SHORTEST_pitfall',
    #                         verbose=args.verbose))
    # agents.append(CBSAgent(0, args.goals[0], belief_update=True, sample_eval=100, verbose=True))
    # agents.append(UniformTreeSearchAgent(0, args.goals[0],
    #                                      belief_update=True, soft_update=2e-4,
    #                                      depth=1, node_eval='CBS',
    #                                      sample_eval=5,
    #                                      sample_backup=10,
    #                                      verbose=True))
    # agents.append(AsymmetricTreeSearch(0, args.goals[0],
    #                                    belief_update=True, verbose=True, soft_update=8e-5,
    #                                    max_it=50,
    #                                    node_eval='CBS',
    #                                    sample_eval=5,
    #                                    sample_select=10,
    #                                    pUCB=False))
    # agents.append(ChasingAgent(0, args.goals[args.agents[0]], p_chase=0.5))  # need to specify stop_on
    # agents.append(SafeAgent(0, args.goals[0]))
    # agents.append(EnhancedSafeAgent(0, args.goals[0]))
    # agents.append(RandomAgent(0, args.goals[args.agents[0]]))
    # agents.append(DijkstraAgent(0, args.goals[args.agents[0]]))
    # agents.append(MDPAgent(0, args.goals[args.agents[0]], belief_update=True, verbose=True))
    # nn_rewards = {
    #     'illegal': 1,
    #     'normal': 1,
    #     'collision': 3000,
    #     'goal': 10
    # }
    # agents.append(UniformTreeSearchAgent(0, args.goals[0],
    #                                      belief_update=True,
    #                                      depth=2, node_eval='NN',
    #                                      verbose=True,
    #                                      nn_estimator=meta_policy,
    #                                      reward_scheme=nn_rewards))

    # agents.append(AStarAgent(1, args.goals[1]))
    # agents.append(RandomAgent(1, args.goals[1], p=0.8))
    # agents.append(DijkstraAgent(1, args.goals[1]))
    # agents.append(SafeAgent(1, args.goals[1]))
    # agents.append(POMDPAgent(1, args.goals[args.agents[1]], exist_policy=True))
    # agents.append(MDPAgent(1, args.goals[1], belief_update=False, verbose=True))
    # agents.append(MetaAgent(1, args.goals[args.agents[1]], meta_policy, belief_update=True, verbose=True))
    # agents.append(QMDPAgent(1, args.goals[args.agents[1]]))
    # agents.append(HistoryMDPAgent(1, args.goals[args.agents[1]], horizon=4))
    # agents.append(UniformTreeSearchAgent(1, args.goals[args.agents[1]],
    #                                      belief_update=True, depth=2, node_eval='HEU-C',
    #                                      verbose=False,
    #                                      check_repeated_states=True))
    # nn_rewards = {
    #     'illegal': 1,
    #     'normal': 1,
    #     'collision': 3000,
    #     'goal': 10
    # }
    # agents.append(UniformTreeSearchAgent(1, args.goals[args.agents[1]],
    #                                      belief_update=True,
    #                                      depth=2, node_eval='NN',
    #                                      verbose=True,
    #                                      nn_estimator=meta_policy,
    #                                      reward_scheme=nn_rewards))
    # agents.append(UniformTreeSearchAgent(1, args.goals[1],
    #                                      belief_update=True,
    #                                      depth=2, node_eval='CBS',
    #                                      sample_eval=10,
    #                                      sample_backup=0,
    #                                      verbose=True))
    # agents.append(CBSAgent(1, args.goals[1], soft_update=2e-5, verbose=True))
    # agents.append(AsymmetricTreeSearch(1, args.goals[1],
    #                                    belief_update=True, verbose=True,
    #                                    max_it=3e2,
    #                                    node_eval='CBS',
    #                                    sample_eval=1,
    #                                    sample_select=50,
    #                                    pUCB=True))
    # agents.append(SafeAgent(2, args.goals[2]))
    # agents.append(AStarAgent(2, args.goals[2]))
    # agents.append(RandomAgent(2, args.goals[2], p=0.6))
    # agents.append(RandomAgent(3, args.goals[3], p=0.8))
    # agents.append(SafeAgent(3, args.goals[3]))

    for i in range(2, args.agents):
        agents.append(AStarAgent(i, args.goals[i]))

    game = MAPF(agents,
                args.starts[:args.agents],
                args.goals[:args.agents],
                args.map,
                randomness=0.,
                stop_on=None)
    history, steps, collisions, stuck = game.run()
    show_hist(history)
    print(steps, collisions, stuck)

    if args.vis:
        paths = []
        for step in history:
            paths.append(step[1])
        if max(args.map.shape) in range(10):
            FPS = 60
        elif max(args.map.shape) in range(10, 15):
            FPS = 30
        else:
            FPS = 15
        animator = Animation(range(args.agents),
                             args.map,
                             args.starts[:args.agents],
                             args.goals[:args.agents],
                             paths,
                             FPS=FPS)
        animator.show()
        if args.save:
            animator.save(file_name=args.save, speed=100)
