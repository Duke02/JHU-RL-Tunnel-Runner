"""
This is the main for like all of our models. Yeehaw.
"""

import typing as tp
from pathlib import Path

from tqdm import tqdm
import numpy as np

from models.tmay9_qagent import QAgent
from RL_Final_Project_agentS import SARSA_0
from MC_agent import RlAgent as MonteCarlo
from environment import Environment as Env

# Total number of episodes to train each agent on.
NUM_EPISODES: int = 10_000
# Total number of agents per AI Type.
NUM_AGENTS: int = 10


DATA_DIR: Path = Path('.').resolve() / 'data'
MODELS_DIR: Path = DATA_DIR / 'models'

def train_qagent(agent: QAgent, env: Env) -> QAgent:
    for _ in tqdm(range(NUM_EPISODES), 'Training QAgent...'):
        agent.reset()
        curr_state: int = env.reset()
        reward: float = 0
        is_terminal: bool = curr_state in env.get_terminal_states()

        while not is_terminal:
            action: int = agent.react_to_state(curr_state, reward)
            curr_state, reward, is_terminal = env.execute_action(action)
        agent.react_to_state(curr_state, reward)
    return agent


def train_sarsa(agent: SARSA_0, env: Env) -> SARSA_0:
    for _ in tqdm(range(NUM_EPISODES), 'Training SARSA...'):
        agent.start_episode()
        curr_state: int = env.reset()
        is_terminal: bool = curr_state in env.get_terminal_states()
        reward: float = 0

        while not is_terminal:
            action: int = agent.select_action(curr_state)
            next_state, reward, is_terminal = env.execute_action(action)

            if is_terminal:
                break

            next_action: int = agent.select_action(next_state)
            agent.update_Q_SARSA(curr_state, action, reward, next_state, next_action)
            curr_state = next_state

    return agent


def train_monte_carlo(agent: MonteCarlo, env: Env) -> MonteCarlo:
    agent.reset()

    for _ in tqdm(range(NUM_EPISODES), 'Training Monte Carlo...'):
        curr_state: int = env.reset()
        agent.new_episode()
        is_terminal: bool = curr_state in env.get_terminal_states()
        reward: float = 0

        while not is_terminal:
            action: int = agent.select_action(curr_state)
            new_state, reward, is_terminal = env.execute_action(action)
            agent.update_episode(curr_state, action, reward)
            curr_state = new_state

    return agent


if __name__ == '__main__':
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=False, exist_ok=True)

    environment: Env = Env()
    qagents: list[QAgent] = [
        QAgent(environment.number_of_states, environment.number_of_possible_actions, initial_epsilon=.1, discount_factor=.9, learning_rate=.1,
               terminal_states=environment.terminal_states, win_states=environment.goal_states, final_epsilon=.1, epsilon_step=0, initial_q_value=0) for _ in range(NUM_AGENTS)]
    sarsas: list[SARSA_0] = [SARSA_0() for _ in range(NUM_AGENTS)]
    monte_carlos: list[MonteCarlo] = [MonteCarlo() for _ in range(NUM_AGENTS)]

    qagents = [train_qagent(q, environment) for q in qagents]
    sarsas = [train_sarsa(s, environment) for s in sarsas]
    monte_carlos = [train_monte_carlo(mc, environment) for mc in monte_carlos]

    qagent_dir: Path = MODELS_DIR / 'QAgent'
    sarsa_dir: Path = MODELS_DIR / 'SARSA'
    mc_dir: Path = MODELS_DIR / 'MonteCarlo'

    qagent_dir.mkdir(exist_ok=True)
    sarsa_dir.mkdir(exist_ok=True)
    mc_dir.mkdir(exist_ok=True)

    for i, (q, s, mc) in enumerate(zip(qagents, sarsas, monte_carlos)):
        q: QAgent
        s: SARSA_0
        mc: MonteCarlo
        print(f'Saving {i}-th model...')
        np.savetxt(qagent_dir / f'model-{i}.mdl', q.qtable)
        np.savetxt(sarsa_dir / f'model-{i}.mdl', s.q)
        np.savetxt(mc_dir / f'model-{i}.mdl', mc.q)

    print('Done!')
