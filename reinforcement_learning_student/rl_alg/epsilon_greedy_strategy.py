from rl_base import ActionControl
from rl_alg.q_agent import QAgent
import numpy as np
from random import *


class EpsilonGreedyStrategy(ActionControl):

    def __init__(self, eps_start, eps_end, eps_dec):
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec

    def update_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon = self.eps_min

    # TODO zaimplementuj strategię eps-zachłanną wyboru akcji
    def get_action(self, agent: QAgent, observation):
        random_nb = random()
        ret = choice(agent.action_space) if random_nb < self.epsilon else np.argmax(agent.q_table[observation, :])
        self.update_epsilon()
        return ret

    def get_instruction_string(self):
        return [f"Automatic eps-greedy."]
