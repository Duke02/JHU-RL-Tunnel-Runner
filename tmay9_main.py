import typing as tp

from models.tmay9_qagent import QAgent
from environment import Environment


n_episodes: int = 100


def train_agent(env: Environment, agent: QAgent) -> QAgent:
    for _ in range(n_episodes):
        agent.reset()
        curr_state: int = env.reset()
        reward: float = 0
        is_terminal: bool = curr_state in env.get_terminal_states()

        while not is_terminal:
            action: int = agent.react_to_state(curr_state, reward)
            curr_state, reward, is_terminal = env.execute_action(action)
        agent.react_to_state(curr_state, reward)
    return agent


if __name__ == '__main__':
    env: Environment = Environment()
    num_agents: int = 10

    best_initial_epsilon: float = 0.05
    best_final_epsilon: float = 0.05
    best_episodes_percent: float = .1
    _num_episodes: int = int(n_episodes * best_episodes_percent)
    best_epsilon_step: float = (best_initial_epsilon - best_final_epsilon) / _num_episodes
    best_starting_q_value: float = -1
    best_discount_factor: float = 0.85
    best_learning_rate_q: float = 0.75
    best_learning_rate_s: float = 0.7

    qlearners: tp.List[QAgent] = [
        QAgent(env.get_number_of_states(), env.get_number_of_actions(), initial_epsilon=best_initial_epsilon, final_epsilon=best_final_epsilon,
               epsilon_step=best_epsilon_step, discount_factor=best_discount_factor, learning_rate=best_learning_rate_q,
               terminal_states=env.get_terminal_states(), initial_q_value=best_starting_q_value) for _ in
        range(num_agents)]
    qlearners = [train_agent(env, agent) for i, agent in enumerate(qlearners)]
