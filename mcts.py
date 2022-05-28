import math
import copy

import numpy as np
import time

import gym_tictactoe
import gym

import argparse
import logging
import sys

C = math.sqrt(2)


def player_swap(player):
    return -1 if player == 1 else 1


class Node:
    def __init__(self, player, parent):
        self.player = player # the player that lead to this state
        self.visit_count = 0
        self.winnings = 0
        self.children = {}
        self.parent = parent

    def select_action(self):
        ucb_scores = {}
        for move, child_node in self.children.items():
            denominator = child_node.visit_count + 1

            ucb_scores.update({move: (child_node.winnings) / denominator + \
                                     C * math.sqrt(math.log(self.visit_count + 1) / denominator)})

        return max(ucb_scores, key=ucb_scores.get, default=None)

    def select_real_action(self):
        scores = {}
        for move, child_node in self.children.items():
            denominator = child_node.visit_count
            scores.update({move: child_node.winnings / denominator})

        return max(scores, key=scores.get, default=None)

    # return leaf_node
    def selection(self):
        node = self
        action = node.select_action()

        while action:
            node = node.children[action]
            action = node.select_action()

        return node

    # actually we make a pseudo expand. We add an empty child. Just for simplification of ucb_scores calculation
    def expand(self, env):
        self.children = {action: Node(player_swap(self.player), self) for action in get_correct_actions(env.state_vector)}

    # update winnings and a total count along the tree (from leaves to root)
    def backpropagate(self, value):
        if value > 0:
            self.winnings += 1
        elif value == 0:
            self.winnings += 0.5

        self.visit_count += 1

        if self.parent:
            self.parent.backpropagate(-value)


def determine_the_winner(player, reward):
    if reward == 10:
        winner = 0
    else:
        winner = player

    if reward < 0:
        raise Exception()

    return winner


# return rollout result for this start_player:
# +1 start_player wins, -1 loose, 0 draw
def rollout(env, start_player):
    done = False
    player = start_player
    while not done:
        action = random_correct_action(env.state_vector)
        state, reward, done, infos = env.step(action, player)

        player = player_swap(player)

    player = player_swap(player)

    return player, reward


def print_tree(node, tab=0, max_deep=1):
    if tab == 0:
        print('root', f'({node.winnings}/{node.visit_count})', f'{node.player}')

    for action, child in node.children.items():
        if child.visit_count:
            print(f'\t' * tab + '├', action, f'({child.winnings}/{child.visit_count})\
             {child.winnings / child.visit_count:.3f}', f'{child.player}')
            if max_deep is False or tab + 1 < max_deep:
                print_tree(child, tab + 1, max_deep=max_deep)


def get_correct_actions(env_state):
    return np.where(np.array(env_state) == 0)[0]


def random_correct_action(env_state):
    possible_actions = get_correct_actions(env_state)
    return np.random.choice(possible_actions)


def human_move(env_state):
    logging.info(env_state)
    possible_actions = get_correct_actions(env_state)
    step = int(input(f'Your move {possible_actions}'))
    return step


def run_mcts(env, player_m, number_of_games, verbose):
    root = Node(player_swap(player_m), None)

    for i in range(number_of_games):

        work_env = copy.deepcopy(env)
        node = root

        # until leaf is reached
        action = node.select_action()
        done = False

        while action is not None and done is not True:
            state, reward, done, _ = work_env.step(action, player_swap(node.player))
            node = node.children[action]

            action = node.select_action()

        if not done:
            node.expand(work_env)

            player, reward = rollout(work_env, player_swap(node.player))
            winner = determine_the_winner(player, reward)
        else:
            winner = determine_the_winner(node.player, reward)

        if winner == 0:
            value = 0
        elif winner == node.player:
            value = 1
        else:
            value = -1

        node.backpropagate(value)

    if verbose:
        print_tree(root, max_deep=1)

    return root.select_real_action()


def game(opponent_move, env, number_of_games, verbose=False, first_player='MCTS'):
    env.reset()
    done = False
    if first_player == 'MCTS':
        player = -1
    elif first_player == 'opponent':
        player = 1

    while not done:
        logging.info(''.join(['='* 20, 'STEP', '=' * 50]))
        if player == -1:
            # MCTS turn
            start_time = time.time()
            action = run_mcts(env, player, number_of_games, verbose)
            end_time = time.time()
            print(f'Elapsed time: {(end_time - start_time) * 1000} ms')

            logging.info(f'1 st player {action}')
        else:
            action = opponent_move(env.state_vector)
            logging.info(f'2 nd player {action}')

        state, reward, done, infos = env.step(action, player)
        logging.info(env.render())
        player = player_swap(player)

    player = player_swap(player)

    logging.info(f'Final_reward: {reward}, Player: {player}')
    if reward == 10:
        print(f'Draw !')
    elif reward == 20:
        if player == -1:
            print(f'X wins')
        elif player == 1:
            print(f'O wins')


def game_against_random(env, first_player, number_of_games, verbose):
    game(random_correct_action, env, number_of_games=number_of_games, first_player=first_player, verbose=verbose)


def game_against_human(env, first_player, number_of_games, verbose):
    game(human_move, env, number_of_games=number_of_games, first_player=first_player, verbose=verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--board', type=int, default=3)
    parser.add_argument('-w', '--win', type=int, default=3, help='minimum winning length')
    parser.add_argument('-f', '--first', action='store_true', help='AI first move')
    parser.add_argument('-m', '--mcts', type=int, default=1000, help='Number of MCTS iterations')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args(sys.argv[1:])

    env = gym.make('TicTacToe-v1', symbols=[-1, 1], board_size=args.board, win_size=args.win)

    if args.first:
        game_against_human(env, 'MCTS', number_of_games=args.mcts, verbose=args.verbose)
    else:
        game_against_human(env, 'opponent', number_of_games=args.mcts, verbose=args.verbose)
