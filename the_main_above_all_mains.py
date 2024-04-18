"""
This is the main for like all of our models. Yeehaw.
"""
import typing as tp
from pathlib import Path
from collections import deque
import json
from copy import deepcopy
from itertools import pairwise

from tqdm import tqdm
import numpy as np
import pandas as pd

from models.tmay9_qagent import QAgent
from models.RL_agentS0 import SARSA_0
from models.MC_agent import RlAgent as MonteCarlo
from Trystan_Files.Project2_agent import RLAgent as TrystanMC
from environment import Environment as Env

# Total number of episodes to train each agent on.
NUM_EPISODES: int = 50_000
# Total number of agents per AI Type.
NUM_AGENTS: int = 10
OPTIMAL_PATH_LENGTH: int = 11
CONVERGENCE_NUM: int = 10

DATA_DIR: Path = Path('.').resolve() / 'data'
MODELS_DIR: Path = DATA_DIR / 'models'


class EpisodicResult(tp.TypedDict):
    perc_states_visited: float
    total_gain: float
    win_rate: float
    did_win: bool
    path_length: int


class TrainingResult(tp.TypedDict):
    agent: QAgent | SARSA_0 | MonteCarlo | None | TrystanMC
    seed: int
    perc_states_visited: float
    total_gain: float
    win_rate: float
    convergence_rate: float
    sample_efficiency: float
    asymptotic_performance: float
    has_converged: bool
    episodic_results: list[EpisodicResult] | None


def has_converged(path_lens: tp.Iterable[int], tolerance: int = 1) -> bool:
    return all(int(abs(pl - OPTIMAL_PATH_LENGTH)) <= tolerance for pl in path_lens)


def train_qagent(agent: QAgent, env: Env) -> TrainingResult:
    total_gain: float = 0
    convergence_rate: int = -1
    path_len_last_n_episodes: deque[int] = deque([], CONVERGENCE_NUM)
    episodic_results: list[EpisodicResult] = []
    num_wins: int = 0
    for episode in tqdm(range(NUM_EPISODES), 'Training QAgent...'):
        if episode > 30:
            converged: bool = has_converged(path_len_last_n_episodes)
            if convergence_rate < 0 and converged:
                convergence_rate = episode
            # This is for if we wanted the first time we "converge" to be
            # not considered the convergence rate.
            # We consider the first convergence to be a true convergence
            # as our models inherently are random and therefore will always
            # have a possibility of not choosing the optimal action at any
            # given state.

            #     if first_convergence_rate < 0:
            #         first_convergence_rate = convergence_rate
            # elif convergence_rate > 0 and not converged:
            #     convergence_rate = -1
            agent.convergence_rate = convergence_rate
        agent.reset()
        curr_state: int = env.reset()
        reward: float = 0
        is_terminal: bool = curr_state in env.get_terminal_states()

        curr_path_len: int = 0
        while not is_terminal:
            action: int = agent.react_to_state(curr_state, reward)
            curr_state, reward, is_terminal = env.execute_action(action)
            total_gain += reward * agent.discount_factor
            curr_path_len += 1
        agent.react_to_state(curr_state, reward)
        did_win: bool = agent.is_win(curr_state)
        if did_win:
            num_wins += 1
        episodic_results.append(EpisodicResult(perc_states_visited=agent.perc_states_visited,
                                               total_gain=total_gain,
                                               win_rate=num_wins / (episode + 1),
                                               did_win=did_win,
                                               path_length=curr_path_len))
        path_len_last_n_episodes.append(curr_path_len if reward > 0 else OPTIMAL_PATH_LENGTH ** 2)

    return TrainingResult(agent=agent,
                          win_rate=float(sum(agent.did_win_last_30)) / min(30, NUM_EPISODES),
                          perc_states_visited=agent.perc_states_visited,
                          total_gain=total_gain,
                          convergence_rate=convergence_rate,
                          sample_efficiency=agent.sample_efficiency,
                          asymptotic_performance=agent.asymptotic_performance,
                          has_converged=convergence_rate > 0,
                          seed=agent.seed,
                          episodic_results=episodic_results)


def train_sarsa(agent: SARSA_0, env: Env) -> TrainingResult:
    total_gain: float = 0
    path_len_last_n_episodes: deque[int] = deque([], CONVERGENCE_NUM)
    states_visited: set[int] = set()
    episodic_results: list[EpisodicResult] = []
    num_wins: int = 0

    for episode in tqdm(range(NUM_EPISODES), 'Training SARSA...'):
        agent.start_episode()
        curr_state: int = env.reset()
        states_visited.add(curr_state)
        is_terminal: bool = curr_state in env.get_terminal_states()

        if not is_terminal:
            action: int = agent.select_action(curr_state)

        curr_path_len: int = 0
        episode_gain: float = 0  # To track gain per episode
        while not is_terminal:
            next_state, reward, is_terminal = env.execute_action(action)
            states_visited.add(next_state)
            next_action: int = agent.select_action(
                next_state) if not is_terminal else None  # SARSA requires next action, which is None if state is terminal
            agent.update_Q_SARSA(curr_state, action, reward, next_state, next_action)
            curr_state, action = next_state, next_action  # Update state and action
            curr_path_len += 1
            episode_gain += reward * (agent.gamma ** curr_path_len)  # Apply discount factor

        path_len_last_n_episodes.append(curr_path_len if reward > 0 else OPTIMAL_PATH_LENGTH ** 2)
        total_gain += episode_gain  # Accumulate episode gain to total gain

        did_win: bool = curr_state in env.goal_states
        if did_win:
            num_wins += 1
        episodic_results.append(EpisodicResult(perc_states_visited=len(states_visited) / env.number_of_states,
                                               total_gain=total_gain,
                                               win_rate=num_wins / (episode + 1),
                                               did_win=did_win,
                                               path_length=curr_path_len))

        agent.convergence_rate = episode if has_converged(
            path_len_last_n_episodes) and agent.convergence_rate < 0 else agent.convergence_rate

    win_rate = num_wins / NUM_EPISODES

    return TrainingResult(agent=agent,
                          total_gain=total_gain,
                          win_rate=win_rate,
                          perc_states_visited=len(states_visited) / agent.get_number_of_states(),
                          convergence_rate=agent.convergence_rate,
                          sample_efficiency=agent.sample_efficiency,
                          asymptotic_performance=agent.asymptotic_performance,
                          has_converged=agent.is_converged,
                          seed=agent.seed,
                          episodic_results=episodic_results)


def train_monte_carlo(agent: MonteCarlo, env: Env) -> TrainingResult:
    agent.reset()

    total_gain: float = 0
    states_visited: set[int] = set()
    path_lens_last_n_episodes: deque[int] = deque([], CONVERGENCE_NUM)

    num_wins: int = 0
    episodic_results: list[EpisodicResult] = []

    for episode in tqdm(range(NUM_EPISODES), 'Training Monte Carlo...'):
        if episode > 30 and not agent.is_converged and has_converged(path_lens_last_n_episodes):
            agent.convergence_rate = episode

        curr_state: int = env.reset()
        agent.new_episode()
        is_terminal: bool = curr_state in env.get_terminal_states()
        reward: float = 0

        curr_path_len: int = 0

        while not is_terminal:
            action: int = agent.select_action(curr_state)
            new_state, reward, is_terminal = env.execute_action(action)
            agent.update_episode(curr_state, action, reward)
            curr_state = new_state
            curr_path_len += 1

        did_win: bool = curr_state in env.goal_states

        if did_win:
            num_wins += 1

        path_lens_last_n_episodes.append(curr_path_len if reward > 0 else OPTIMAL_PATH_LENGTH ** 2)

        curr_gain, _ = agent.update_q()
        total_gain += curr_gain
        states_visited.update(set(agent.visited_states))
        episodic_results.append(EpisodicResult(perc_states_visited=len(states_visited) / env.number_of_states,
                                               total_gain=total_gain,
                                               win_rate=num_wins / (episode + 1),
                                               did_win=did_win,
                                               path_length=curr_path_len))

    return TrainingResult(agent=agent,
                          total_gain=total_gain,
                          perc_states_visited=len(states_visited) / env.number_of_states,
                          win_rate=num_wins / NUM_EPISODES,
                          convergence_rate=agent.convergence_rate,
                          sample_efficiency=agent.sample_efficiency,
                          asymptotic_performance=agent.asymptotic_performance,
                          has_converged=agent.is_converged,
                          seed=agent.seed,
                          episodic_results=episodic_results)


def create_training_result_to_save(tr: TrainingResult) -> TrainingResult:
    out: TrainingResult = deepcopy(tr)
    out['agent'] = None
    out['episodic_results'] = None
    return out


if __name__ == '__main__':
    print('Initializing...')
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=False, exist_ok=True)

    environment: Env = Env()
    qagents: list[QAgent] = [
        QAgent(environment.number_of_states, environment.number_of_possible_actions, initial_epsilon=0.1,
               discount_factor=.9, learning_rate=.1,
               terminal_states=environment.terminal_states, win_states=environment.goal_states, final_epsilon=.1,
               epsilon_step=0, initial_q_value=0, seed=13 * i) for i in range(NUM_AGENTS)]
    sarsas: list[SARSA_0] = [SARSA_0(seed=13 * i) for i in
                             range(NUM_AGENTS)]
    monte_carlos: list[MonteCarlo | TrystanMC] = [MonteCarlo(seed=13 * i) for i in range(NUM_AGENTS)]
    # [TrystanMC(num_actions=environment.number_of_possible_actions, num_states=environment.number_of_states,
    #                                        epsilon=.1, seed=13 * i) for i in range(NUM_AGENTS)]

    print('Training models...')

    qagent_results: list[TrainingResult] = [train_qagent(q, environment) for q in qagents]
    sarsa_results: list[TrainingResult] = [train_sarsa(s, environment) for s in sarsas]
    mc_results: list[TrainingResult] = [train_monte_carlo(mc, environment) for mc in monte_carlos]

    qagent_dir: Path = MODELS_DIR / 'QAgent'
    sarsa_dir: Path = MODELS_DIR / 'SARSA'
    mc_dir: Path = MODELS_DIR / 'MonteCarlo'

    qagent_dir.mkdir(exist_ok=True)
    sarsa_dir.mkdir(exist_ok=True)
    mc_dir.mkdir(exist_ok=True)

    print('Saving models...')

    for i, (qr, sr, mcr) in enumerate(zip(qagent_results, sarsa_results, mc_results)):
        qr: TrainingResult
        sr: TrainingResult
        mcr: TrainingResult
        q: QAgent = qr['agent']
        s: SARSA_0 = sr['agent']
        mc: MonteCarlo = mcr['agent']
        np.savetxt(qagent_dir / f'model-{i}.mdl', q.qtable)
        np.savetxt(sarsa_dir / f'model-{i}.mdl', s.q)
        np.savetxt(mc_dir / f'model-{i}.mdl', mc.q)

        qrdf: pd.DataFrame = pd.DataFrame.from_records(qr['episodic_results'])
        qrdf.to_csv(qagent_dir / f'results-QAgent-{i}.csv', index=False)
        srdf: pd.DataFrame = pd.DataFrame.from_records(sr['episodic_results'])
        srdf.to_csv(sarsa_dir / f'results-SARSA-{i}.csv', index=False)
        mcrdf: pd.DataFrame = pd.DataFrame.from_records(mcr['episodic_results'])
        mcrdf.to_csv(mc_dir / f'results-MonteCarlo-{i}.csv', index=False)

    print('Processing results...')

    q_results_to_save: list[TrainingResult] = [create_training_result_to_save(tr) for tr in qagent_results]
    sarsa_results_to_save: list[TrainingResult] = [create_training_result_to_save(tr) for tr in sarsa_results]
    mc_results_to_save: list[TrainingResult] = [create_training_result_to_save(tr) for tr in mc_results]

    q_df: pd.DataFrame = pd.DataFrame.from_records(q_results_to_save)
    q_df['agent'] = 'QAgent'
    s_df: pd.DataFrame = pd.DataFrame.from_records(sarsa_results_to_save)
    s_df['agent'] = 'SARSA_0'
    mc_df: pd.DataFrame = pd.DataFrame.from_records(mc_results_to_save)
    mc_df['agent'] = 'MonteCarlo'

    print('Saving results...')

    results_df: pd.DataFrame = pd.concat([q_df, s_df, mc_df])
    results_df.to_csv(MODELS_DIR / 'results.csv', index=False)

    print('Done!')
