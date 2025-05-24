
# Deep Reinforcement Learning

This project explores various Deep Reinforcement Learning algorithms applied to custom game environments like Farkle, TicTacToe, GridWorld, all of which were implemented from scratch. The goal is to implement, train, and compare different DRL agents to evaluate their performance in strategic decision-making environments.

## 🚀 Project Goals
- Implement classic and modern DRL algorithms such as DQN, DDQN, PPO, REINFORCE, Actor-Critic, and more.
- Develop environments for different games to test the performance of agents.
- Compare agent strategies and learning behaviors across multiple environments.
- Provide a modular and extendable framework for DRL experimentation.

## 📦 Installation
1. Clone the Repository

```bash
git clone https://github.com/yourusername/DeepReinforcementLearningGames.git
cd DeepReinforcementLearningGames
```

2. Create a Virtual Environment (Optional but recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install Dependencies

```bash
pip install -r requirements.txt

## With UV
uv sync
```

## 🧠 How to Run
The entry point is `main.py`, which can be configured to run different agents in various environments.

### Example:
```bash
python main.py
```
Uncomment and change the following:

`env`: Specify the DRL environment

`n_episode`: Specify the numbers of episode on which the agent will train.

All functions to train with a specific algorithms are implemented in the `main.py` you just have to chose.

You can customize further by editing main.py.



## 📁 Project Structure
```
.
├── agent/                          # All DRL algorithms, all future agents should be implemented here
|   ├── Farkle/                     # Some custom Farkle DRl algorithms
|       ├── ActorCritic.py
|       └── ...
│   ├── DQN.py                      # Deep Q-Network
│   ├── DDQN.py                     # Double DQN
│   ├── PPO.py                      # Proximal Policy Optimization
│   ├── Reinforce.py                # REINFORCE algorithm
│   ├── ActorCritic.py              # Actor-Critic agent
│   ├── replay_buffer.py            # Experience replay buffer
│   └── ...                         # Other agents and support files
│
├── environment/                    # Game environments, all futur envrionment should be implemented here
│   ├── Farkle.py                   # Farkle game logic
│   ├── TicTacToe.py                # Tic Tac Toe game logic
│   ├── GridWorld.py                # Grid-based environment
│   ├── base.py                     # Base Interface                   
│
├── results/                        # Logs and result outputs per agent
│   ├── dqn/                        # Logs and result of DQN training on different environment
|       ├── Farkle                  
|           ├── result.json         # JSON containing the rewards per step and other metrics
|           ├── training_plots.jpg  # An image with multiple plots to see how to agent perform during training
|       ├── LineWorld/
|           └── ...
|       ├── GridWorld/
|           └── ...
|       ├── TicTacToe/
|           └── ...
│   ├── ppo/
|       └── ...
│   ├── reinforce/
|       └── ...
│   └── ...
│
├── tools/                    # Utility scripts
│   └── utils.py              # Helper functions
│
├── main.py                   # Main script to run experiments
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 📊 Results & Evaluation
Training results are stored in the `results/` directory, categorized by agent type. Each subdirectory contains performance logs, and potentially training plots.
Models and models checkpoint are saved in a `models/` directory but the folder will not be uploaded to GitHub as it is too heavy.

## Helpful Links
- [Deep Reinforcement Learning Hands-On (Book)](https://www.packtpub.com/en-us/product/deep-reinforcement-learning-hands-on-9781838826994)
- [CS285: Deep Reinforcement Learning (Berkeley)](https://rail.eecs.berkeley.edu/deeprlcourse/)

