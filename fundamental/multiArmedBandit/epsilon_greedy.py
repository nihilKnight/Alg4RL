import numpy as np
from solver import Solver


class EpsilonGreedy(Solver):
    """ epsilon贪婪算法,继承Solver类 """
    # 乐观初始化法（Optimistic Initial Values）加速收敛。
    # 将初始奖励期望估值设定为1.0，使每个拉杆看起来都很“好”，算法会主动多试几次，起到增加探索的作用。
    # 在 epsilon 较小或完全贪婪时有作用，因为它避免了“第一次尝试差臂后就永远不碰”的问题。
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        # 初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)  # 随机选择一根拉杆
        else:
            k = np.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆
        r = self.bandit.step(k)  # 得到本次动作的奖励
        print(self.estimates, k, r, end='\t\t')
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        print(self.estimates[k])
        return k


class DecayingEpsilonGreedy(Solver):
    """ epsilon值随时间衰减的epsilon-贪婪算法,继承Solver类 """
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:  # epsilon值随时间衰减，epsilon = 1 / t
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])

        return k