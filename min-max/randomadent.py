import math
import copy
from itertools import permutations

from exceptions import AgentException


class RandomAgent:
    def __init__(self, my_token='o', rival_token='x'):
        self.my_token = my_token
        self.rival_token = rival_token

    def heuristic(self, state):
        score = 3 * state.center_column().count(self.my_token)
        score -= state.center_column().count(self.rival_token)
        for window in state.iter_fours():
            if window in permutations([self.my_token, self.my_token, self.my_token, "_"]):
                score += 5
            elif window in permutations([self.my_token, self.my_token, "_", "_"]):
                score += 2
            elif window in permutations([self.rival_token, self.rival_token, "_", "_"]):
                score -= 2
            elif window in permutations([self.rival_token, self.rival_token, self.rival_token, "_"]):
                score -= 5
        return score

    def minmax(self, connect4, depth, alpha, beta, maximizing_player):
        if depth == 0:
            return self.heuristic(connect4), 0
        elif connect4.game_over:
            if self.my_token == connect4.wins:
                return 10000, 0
            elif connect4.wins is None:
                return 0, 0
            else:
                return -10000, 0
        elif maximizing_player:
            max_evaluation = -1 * math.inf
            best_move = None

            for drop in connect4.possible_drops():
                token_copy = copy.deepcopy(connect4)
                token_copy.drop_token(drop)

                (evaluation, _) = self.minmax(token_copy, depth - 1, alpha, beta, False)
                if evaluation > max_evaluation:
                    max_evaluation = evaluation
                    best_move = drop

                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break

            return max_evaluation, best_move
        else:
            min_evaluation = math.inf
            best_move = None

            for drop in connect4.possible_drops():
                token_copy = copy.deepcopy(connect4)
                token_copy.drop_token(drop)

                (evaluation, _) = self.minmax(token_copy, depth - 1, alpha, beta, True)
                if evaluation < min_evaluation:
                    min_evaluation = evaluation
                    best_move = drop

                beta = min(beta, evaluation)
                if beta <= alpha:
                    break

            return min_evaluation, best_move

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        (result, move) = self.minmax(connect4, 6, -1 * math.inf, math.inf, True)
        return connect4.drop_token(move)
