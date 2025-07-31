import numpy as np
import matplotlib.pyplot as plt
from bernoulli_bandit import BernoulliBandit
from epsilon_greedy import EpsilonGreedy, DecayingEpsilonGreedy
from ucb import UCB
from thompson_sampling import ThompsonSampling


def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


np.random.seed(49)  # 设定随机种子,使实验具有可重复性
K = 10
bandit_10_arm = BernoulliBandit(K)
print("随机生成了一个%d臂伯努利老虎机" % K)
print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" %
      (bandit_10_arm.best_idx, bandit_10_arm.best_prob))

solvers = []
solver_names = []

epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.1)
epsilon_greedy_solver.run(5000)
print("Epsilon贪婪算法的累积懊悔为: %.2f" % epsilon_greedy_solver.regret)
solvers.append(epsilon_greedy_solver)
solver_names.append("Epsilon Greedy")

decay_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decay_epsilon_greedy_solver.run(5000)
print("衰减Epsilon贪婪算法的累积懊悔为: %.2f" % decay_epsilon_greedy_solver.regret)
solvers.append(decay_epsilon_greedy_solver)
solver_names.append("Decaying Epsilon Greedy")

ucb = UCB(bandit_10_arm, coef=1)
ucb.run(5000)
print("UCB算法的累积懊悔为: %.2f" % ucb.regret)
solvers.append(ucb)
solver_names.append("UCB")

thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(5000)
print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
solvers.append(thompson_sampling_solver)
solver_names.append("Thompson Sampling")

plot_results(solvers, solver_names)
# plot_results([decay_epsilon_greedy_solver], ["Decaying Epsilon Greedy"])