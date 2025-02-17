from tqdm import tqdm

from environment.LineWorld import LineWorld, play_lineworld
from environment.GridWorld import GridWorld, play_gridworld
from environment.TicTacToe import TicTacToe, play_tictactoe_vs_random
from environment.Farkle import Farkle, play_farkle_vs_random
from agent.random_agent import RandomAgent
from agent.Q_Learning import QLearningAgent
from agent.DQN import DQNAgent
from agent.Farkle.DQN import DQNAgent as FarkleDQNAgent
from agent.DDQN import DDQNAgent, play_game as ddqn_play_game
from agent.DDQNPER import DDQNPERAgent
from agent.Reinforce import ReinforceAgent
from agent.Farkle.Reinforce import ReinforceAgent as FarkleReinforceAgent, play_farkle_reinforce
from agent.ReinforceBaseline import ReinforceBaselineAgent
from agent.Farkle.ReinforceBaseline import ReinforceBaselineAgent as FarkleReinforceBaselineAgent, play_farkle_reinforce_baseline
from agent.ActorCritic import ActorCriticAgent
from agent.Farkle.ActorCritic import ActorCriticAgent as FarkleActorCriticAgent, play_farkle_ac
from agent.PPO import PPOAgent
from agent.Farkle.PPO import PPOAgent as FarklePPOAgent, play_farkle_ppo
from agent.MCTS import play_multiple_games

from tools.utils import calculate_epsilon_decay


def random_agent(env, n_episode, save_dir='results/random_agent/'):
    env = env
    agent = RandomAgent(action_space=env.action_space)

    results = agent.train(
        env=env,
        episodes=n_episode,
        eval_frequency=1000,
        num_eval_episodes=1000,
        save_dir=save_dir
    )

def q_learning(env, n_episode, save_dir='results/q_learning/'):
    env = env
    agent = QLearningAgent(action_space=env.action_space,
                           observation_space=env.observation_space,
                           epsilon_decay=calculate_epsilon_decay(n_episodes=n_episode),)

    results = agent.train(
        env=env,
        episodes=n_episode,
        eval_frequency=1000,
        num_eval_episode=1000,
        save_dir=save_dir
    )


def dqn(env, n_episode, save_dir='results/dqn/'):
    env = env
    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=calculate_epsilon_decay(n_episodes=n_episode),
        memory_size=100000,
        batch_size=64,
        target_update=100
    )

    results = agent.train(
        env=env,
        episodes=n_episode,
        eval_frequency=1000,
        num_eval_episodes=1000,
        save_dir=save_dir
    )


def ddqn(env, n_episode, save_dir='results/ddqn/'):
    env = env
    agent = DDQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=calculate_epsilon_decay(n_episodes=n_episode),
        memory_size=100000,
        batch_size=64,
        target_update=500,
    )

    results = agent.train(
        env=env,
        episodes=n_episode,
        eval_frequency=1000,
        num_eval_episodes=1000,
        save_dir=save_dir
    )


def farkle_dqn(env, n_episode, save_dir='results/farkle_dqn/'):
    env = env
    agent = FarkleDQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=0.0005,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=calculate_epsilon_decay(n_episodes=n_episode),
        memory_size=100000,
        batch_size=64,
        target_update=1000,
    )

    results = agent.train(
        env=env,
        episodes=n_episode,
        eval_frequency=1000,
        num_eval_episodes=1000,
        save_dir=save_dir
    )


def ddqwper(env, n_episode, save_dir='results/ddqwper/'):
    env = env
    agent = DDQNPERAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=0.0005,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=calculate_epsilon_decay(n_episodes=n_episode),
        memory_size=100000,
        batch_size=64,
        target_update=1000,
    )

    results = agent.train(
        env=env,
        episodes=n_episode,
        eval_frequency=1000,
        num_eval_episodes=1000,
        save_dir=save_dir
    )


def reinforce(env, n_episode, save_dir='results/reinforce/'):
    env = env
    agent = ReinforceAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=5e-3,
        gamma=0.99,
    )

    results = agent.train(
        env=env,
        episodes=n_episode,
        eval_frequency=1000,
        num_eval_episodes=1000,
        save_dir=save_dir
    )


def farkle_reinforce(env, n_episode, save_dir='results/reinforce/'):
    env = env
    agent = FarkleReinforceAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=1e-4,
        gamma=0.99,
    )

    results = agent.train(
        env=env,
        episodes=n_episode,
        eval_frequency=1000,
        num_eval_episodes=1000,
        save_dir=save_dir
    )


def reinforce_baseline(env, n_episode, save_dir='results/reinforce_baseline_critic/'):
    env = env
    agent = ReinforceBaselineAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy_lr=0.002,
        value_lr=0.002,
        gamma=0.99,
    )

    results = agent.train(
        env=env,
        episodes=n_episode,
        eval_frequency=1000,
        num_eval_episodes=1000,
        save_dir=save_dir
    )


def farkle_reinforce_baseline(env, n_episode, save_dir='results/reinforce_baseline/'):
    env = env
    agent = FarkleReinforceBaselineAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy_lr=0.0002,
        value_lr=0.002,
        gamma=0.99,
    )

    results = agent.train(
        env=env,
        episodes=n_episode,
        eval_frequency=1000,
        num_eval_episodes=1000,
        save_dir=save_dir
    )


def actor_critic(env, n_episode, save_dir='results/actor_critic/'):
    env = env
    agent = ActorCriticAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate_actor=2e-5,
        learning_rate_critic=1e-5,
        gamma=0.99,
    )

    results = agent.train(
        env=env,
        episodes=n_episode,
        eval_frequency=1000,
        num_eval_episodes=1000,
        save_dir=save_dir
    )


def farkle_actor_critic(env, n_episode, save_dir='results/actor_critic/'):
    env = env
    agent = FarkleActorCriticAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate_actor=0.0005,
        learning_rate_critic=0.0005,
        gamma=0.99,
    )

    results = agent.train(
        env=env,
        episodes=n_episode,
        eval_frequency=1000,
        num_eval_episodes=1000,
        save_dir=save_dir
    )

def ppo(env, n_episode, save_dir='results/ppo/'):
    env = env
    agent = PPOAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=0.0005,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=32
    )

    results = agent.train(
        env=env,
        episodes=n_episode,
        eval_frequency=1000,
        num_eval_episodes=1000,
        save_dir=save_dir
    )


def farkle_ppo(env, n_episode, save_dir='results/ppo/'):
    env = env
    agent = FarklePPOAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=1e-5,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=32
    )

    results = agent.train(
        env=env,
        episodes=n_episode,
        eval_frequency=1000,
        num_eval_episodes=1000,
        save_dir=save_dir
    )


def mcts(env, n_games, save_dir='results/mcts/'):
    env = env

    play_multiple_games(env, 100, save_dir=save_dir)


def demo():
    env = Farkle()
    # play_farkle_ac(env, model_path='models/actor_critic/Farkle/model_best.pth')
    # play_farkle_ppo(env, model_path='models/ppo/Farkle/model_best.pth')
    # play_farkle_reinforce(
    #     env, model_path='models/reinforce/Farkle/model_best.pth')
    play_farkle_reinforce_baseline(
        env, model_path='models/reinforce_baseline/Farkle/model_best.pth')

if __name__ == "__main__":

    # play_tictactoe_vs_random()
    play_farkle_vs_random()
    # demo()
    # env = GridWorld()
    # n_episode = 100000

    random_line_dir = 'results/random_agent/LineWorld/'
    random_grid_dir = 'results/random_agent/GridWorld/'
    random_tic_dir = 'results/random_agent/TicTacToe/'
    random_fark_dir = 'results/random_agent/Farkle/'

    q_learning_line_dir = 'results/q_learning/LineWorld/'
    q_learning_grid_dir = 'results/q_learning/GridWorld/'
    q_learning_tic_dir = 'results/q_learning/TicTacToe/'
    q_learning_fark_dir = 'results/q_learning/Farkle/'

    dqn_line_dir = 'results/dqn/LineWorld/'
    dqn_grid_dir = 'results/dqn/GridWorld/'
    dqn_tic_dir = 'results/dqn/TicTacToe/'
    dqn_fark_dir = 'results/dqn/Farkle/'

    ddqn_line_dir = 'results/ddqn/LineWorld/'
    ddqn_grid_dir = 'results/ddqn/GridWorld/'
    ddqn_tic_dir = 'results/ddqn/TicTacToe/'
    ddqn_fark_dir = 'results/ddqn/Farkle/'

    ddqwper_line_dir = 'results/ddqwper/LineWorld/'
    ddqwper_grid_dir = 'results/ddqwper/GridWorld/'
    ddqwper_tic_dir = 'results/ddqwper/TicTacToe/'
    ddqwper_fark_dir = 'results/ddqwper/Farkle/'

    reinforce_line_dir = 'results/reinforce/LineWorld/'
    reinforce_grid_dir = 'results/reinforce/GridWorld/'
    reinforce_tic_dir = 'results/reinforce/TicTacToe/'
    reinforce_fark_dir = 'results/reinforce/Farkle/'

    reinforce_baseline_line_dir = 'results/reinforce_baseline/LineWorld/'
    reinforce_baseline_grid_dir = 'results/reinforce_baseline/GridWorld/'
    reinforce_baseline_tic_dir = 'results/reinforce_baseline/TicTacToe/'
    reinforce_baseline_fark_dir = 'results/reinforce_baseline/Farkle/'

    actor_critic_line_dir = 'results/actor_critic/LineWorld/'
    actor_critic_grid_dir = 'results/actor_critic/GridWorld/'
    actor_critic_tic_dir = 'results/actor_critic/TicTacToe/'
    actor_critic_fark_dir = 'results/actor_critic/Farkle/'

    ppo_line_dir = 'results/ppo/LineWorld/'
    ppo_grid_dir = 'results/ppo/GridWorld/'
    ppo_tic_dir = 'results/ppo/TicTacToe/'
    ppo_fark_dir = 'results/ppo/Farkle/'

    random_rollout_line_dir = 'results/random_rollout/LineWorld/'
    random_rollout_grid_dir = 'results/random_rollout/GridWorld/'
    random_rollout_tic_dir = 'results/random_rollout/TicTacToe/'
    random_rollout_fark_dir = 'results/random_rollout/Farkle/'

    mcts_line_dir = 'results/mcts/LineWorld/'
    mcts_grid_dir = 'results/mcts/GridWorld/'
    mcts_tic_dir = 'results/mcts/TicTacToe/'

    # random_agent(env, n_episode, save_dir=random_grid_dir)
    # q_learning(env, n_episode, save_dir=q_learning_fark_dir)
    # dqn(env, n_episode, save_dir=dqn_fark_dir)
    # farkle_dqn(env, n_episode, save_dir=dqn_fark_dir)
    # ddqn(env, n_episode, save_dir=ddqn_fark_dir)
    # ddqwper(env, n_episode, save_dir=ddqwper_grid_dir)
    # reinforce(env, n_episode, save_dir=reinforce_fark_dir)
    # farkle_reinforce(env, n_episode, save_dir=reinforce_fark_dir)
    # reinforce_baseline(env, n_episode, save_dir=reinforce_baseline_fark_dir)
    # farkle_reinforce_baseline(
    #     env, n_episode, save_dir=reinforce_baseline_fark_dir)
    # actor_critic(env, n_episode, save_dir=actor_critic_fark_dir)
    # farkle_actor_critic(env, n_episode, save_dir=actor_critic_fark_dir)
    # ppo(env, n_episode, save_dir=ppo_fark_dir)
    # farkle_ppo(env, n_episode, save_dir=ppo_fark_dir)
    # random_rollout(env, n_episode, save_dir=random_rollout_tic_dir)
    # mcts(env, n_episode, save_dir=mcts_line_dir)
