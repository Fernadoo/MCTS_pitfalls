# Pitfalls for MCTS

This repo implements several situations that easily set up Monte Carlo Tree Search (MCTS) algorithms.

<img src="/Users/fernando/Library/CloudStorage/OneDrive-HKUSTConnect/mcts_pitfall/layout.png" alt="layout" style="zoom:15%;" />

> The sphere with a number is the moving robot. The square is her goal. A orange diamond is designed as a trap.

### Quick Start

##### Dependencies

```shell
pip install -r requirements.txt
```

##### Usage

```shell
usage: run.py [-h] [--agents AGENTS] [--map MAP] [--starts STARTS [STARTS ...]] [--goals GOALS [GOALS ...]] [--randomness RANDOMNESS] [--it IT] [--verbose VERBOSE]
              [--vis] [--save SAVE]

Pitfalls for Single-Agent Planning.

optional arguments:
  -h, --help            show this help message and exit
  --agents AGENTS       Specify the number of agents
  --map MAP             Specify a map
  --starts STARTS [STARTS ...]
                        Specify the starts for each agent, e.g., 2_0 0_2, or simple type `random`
  --goals GOALS [GOALS ...]
                        Specify the goals for each agent, e.g., 5_1 6_4, or simple type `random`
  --randomness RANDOMNESS
                        Specify the extent of randomness of the env
  --it IT               Specify the number of iterations for MCTS
  --verbose VERBOSE     Specify the need of showing debug info
  --vis                 Visulize the process
  --save SAVE           Specify the path to save the animation
```

##### A Running Example:

Try the following command,

```shell
python run.py --agents 1 --map square --starts 7_4 --goals 10_6 --randomness 0.1 --it 10 --vis --verbose 2
```

You will get the following results,

1. The planning history

   ```shell
   100%|████████████████████████| 10/10 [00:00<00:00, 5561.26it/s]
   Took 10 simus, lookahead for 2.0 steps
   Shortest dist to goal: 5
   visit counts: [2. 1. 1. 4. 1.]
   action values: [4.08586096       -inf 3.56281643 4.61154713 3.56281643]
   eventually took 3
   
   ...
   
   T0: start from ((7, 4),)
   T1: actions: ['down']	locations: ((8, 4),)
   T2: actions: ['down']	locations: ((9, 4),)
   T3: actions: ['right']	locations: ((9, 5),)
   T4: actions: ['down']	locations: ((10, 5),)
   T5: actions: ['right']	locations: ((10, 6),)
   ```

2. The visualized tree search process

   <img src="/Users/fernando/Library/CloudStorage/OneDrive-HKUSTConnect/mcts_pitfall/tree_visualization.pdf" alt="layout" style="zoom:15%;" />

3. A video showing the eventual pathfinding