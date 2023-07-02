from copy import deepcopy
import numpy as np
import copy # MAKE SURE THIS IS ALLOWED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
from numpy import inf

# function VALUE-ITERATION(mdp, ε) return a utility function
# inputs: mdp, an MDP with states S, actions A(s), transition model P(s’|s,a), rewards R(s), discount γ, ε the maximum change in the utility of any state in an iteration
# local variables: U, U’, vectors of utilities for states in S, initially zero, δ the maximum change in the utility of any state in an iteration
# repeat:
# 〖U=U〗^',δ=0
# for each state s in S do:
#  U^' (s)= R(s)+max_a€A(s) ⁡〖∑_(s^')▒〖P(s^'│s,a)[R(s,a)+γ〗 U(s')]〗
# δ=max⁡(δ,|U^' (s)-U(s)|)
# until δ<ε(1-γ)/γ or (δ=0 and γ=1)
# return U


def max_expectation(mdp, r, c, U):
    max_sum = float(-inf)
    action_policy = 'UP'
    for index, action in enumerate(['UP', 'DOWN', 'RIGHT', 'LEFT']):
        prob = mdp.transition_function[action]
        prob_sum = 0
        for i, next_action in enumerate(['UP', 'DOWN', 'RIGHT', 'LEFT']):
            next_r, next_c = mdp.step((r, c), next_action)
            prob_sum += float(prob[i])*float(U[next_r][next_c])
        if prob_sum > max_sum:
            max_sum = prob_sum
            action_policy = index
    return action_policy, max_sum


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #
    # ====== YOUR CODE: ======
    U = copy.deepcopy(U_init)
    U_tag = copy.deepcopy(U_init)
    while True:
        U = copy.deepcopy(U_tag)
        delta = 0
        for r in range(mdp.num_row):
            for c in range(mdp.num_col):
                reward = mdp.board[r][c]
                if (r, c) in mdp.terminal_states:
                    U_tag[r][c] = reward
                    continue
                elif reward == 'WALL':
                    U_tag[r][c] = None
                    continue
                else:  # valid state with reward
                    reward = round(float(reward), 3)
                    _, curr_expectation = max_expectation(mdp, r, c, U)
                    U_tag[r][c] = reward + mdp.gamma*curr_expectation
                ssss=abs(U_tag[r][c] - U[r][c])
                if abs(U_tag[r][c] - U[r][c]) > delta:
                    delta = abs(U_tag[r][c] - U[r][c])
        if delta == 0 or mdp.gamma == 0 or delta < (epsilon*(1-mdp.gamma))/mdp.gamma:
            break
    return U
    # ========================


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    policy = [[0 for i in range(mdp.num_col)] for i in range(mdp.num_row)]
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            reward = mdp.board[r][c]
            if (r, c) in mdp.terminal_states:
                continue
            elif reward == 'WALL':
                continue
            else:  # valid state with reward
                policy[r][c], _ = max_expectation(mdp, r, c, U)
    return policy
    # ========================


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    # ====== YOUR CODE: ======
    P = [[0 for i in range(mdp.num_col*mdp.num_row)] for i in range(mdp.num_row*mdp.num_col)]
    R = [0 for i in range(mdp.num_col*mdp.num_row)]
    U = [[0 for i in range(mdp.num_col)] for i in range(mdp.num_row)]
    valid_actions = ['UP', 'DOWN', 'RIGHT', 'LEFT']

    for i in range(mdp.num_row*mdp.num_col):
        R[i] = 0

    for r in range(mdp.num_row*mdp.num_col):
        for c in range(mdp.num_row*mdp.num_col):
            P[r][c] = 0


    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            reward = mdp.board[r][c]
            if (r, c) in mdp.terminal_states:
                R[c + mdp.num_col * r] = 0
                continue
            elif reward == 'WALL':
                continue
            else:  # valid state with reward
                R[c+mdp.num_col*r] = float(reward)

                for i, next_action in enumerate(['UP', 'DOWN', 'RIGHT', 'LEFT']):
                    next_r, next_c = mdp.step((r, c), next_action)
                    prev_action = valid_actions[policy[r][c]]
                    P[c+mdp.num_col*r][next_c+mdp.num_col*next_r] = -(mdp.gamma * (mdp.transition_function[prev_action])[i])

    print("P: ", P)
    print("R: ", R)

    print("P dimensions:", len(P), "x", len(P[0]))
    print("R dimensions:", len(R), "x", "1")

    print("P Rank:", np.linalg.matrix_rank(P))

    x = np.linalg.solve(P, R)
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            if (r, c) in mdp.terminal_states:
                U[r][c] = mdp.board[r][c]
            elif mdp.board[r][c] == 'WALL':
                U[r][c] = None
            else:  # valid state with reward
                U[r][c] = x[c+r*mdp.num_col]
    return U
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
