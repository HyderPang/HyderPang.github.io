import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import datetime


def normal(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * 3.1315926)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def gamma(x, mu, sigma):
    alpha = (mu / sigma) ** 2
    beta = alpha / mu
    return st.gamma.pdf(x, alpha, scale=1 / beta)


def trapezium(x, *args):
    if len(args) == 2:
        mu1, sigma = args
        mu2 = mu1
    elif len(args) == 3:
        mu1, mu2, sigma = args
    y = np.zeros_like(x) * 1.0
    y[(x < mu1) & (x >= mu1 - sigma)] = (x[(x < mu1) & (x >= mu1 - sigma)] - mu1) / sigma + 1
    y[(x > mu2) & (x <= mu2 + sigma)] = (mu2 - x[(x > mu2) & (x <= mu2 + sigma)]) / sigma + 1
    y[(mu1 <= x) & (x <= mu2)] = 1
    return y


class Animal:
    def __init__(self, init_date=[2022, 5, 1], number=[1000, 1000], Q_init=None, matting_pattern='monogamy', N_mmp=10,
                 r_stages='default', growth_stages=[2, 4, 12, 15], max_life_month=15, BR=7., BR_m='default',
                 unit_energy_scale=1, E_individual_a='default', w_enerygy_m='default',
                 r_DA_a='default', r_DA_m='default', DA_rate=0.001,
                 r_dem_a='default', r_dem_m='default', r_scale=0.05, P_alloc='default', N_duration=3
                 ):
        self.a_max = max_life_month * 31
        self.a_array = np.arange(self.a_max)
        self.date = datetime.date(init_date[0], init_date[1], init_date[2])
        if Q_init == None:
            self.Q = np.ones([self.a_max, 2])
        else:
            self.Q = Q_init
        self.Q = self.Q * np.array(number) / np.sum(self.Q) * 2

        ## 1.Growth
        self.matting_pattern = matting_pattern
        self.N_mmp = N_mmp
        if r_stages == 'default':
            self.infant_prob = trapezium(self.a_array, 0, growth_stages[0] * 31)
            self.juvenile_prob = trapezium(self.a_array, growth_stages[0] * 31, growth_stages[0] * 31)
            self.senior_prob = trapezium(self.a_array, growth_stages[3] * 31,
                                         (growth_stages[3] - growth_stages[2]) * 31)
            self.adult_prob = 1 - self.infant_prob - self.juvenile_prob - self.senior_prob
        else:
            self.infant_prob, self.adult_prob, self.juvenile_prob, self.senior_prob = r_stages

        if BR_m == 'default':
            self.BR_m = np.array([0.06, 0.08, 0.11, 0.13, 0.13, 0.13, 0.11, 0.08, 0.06, 0.04, 0.04, 0.04, ])
        else:
            self.BR_m = BR_m
        self.BR_m = BR * self.BR_m / np.mean(self.BR_m)

        ## 2.Hunted
        if E_individual_a == 'default':
            E_individual_a = 0.6 * trapezium(self.a_array, growth_stages[0] * 31, growth_stages[3] * 31,
                                             (growth_stages[1] - growth_stages[0]) * 31) \
                             + 0.4 * trapezium(self.a_array, growth_stages[1] * 31, growth_stages[2] * 31,
                                               (growth_stages[2] - growth_stages[1]) * 31)
        self.E_a = E_individual_a / np.max(E_individual_a) * unit_energy_scale

        if w_enerygy_m == 'default':
            w_enerygy_m = np.ones([12, 1])
        self.E_m = w_enerygy_m / np.max(w_enerygy_m)

        # 3.Disease & Aging
        if r_DA_a == 'default':
            r_DA_a = normal(self.a_array, self.a_max, self.a_max / 16) + normal(self.a_array, 0, self.a_max / 16)
        self.r_DA_a = DA_rate / 2 * r_DA_a / np.sum(r_DA_a)

        if r_DA_m == 'default':
            r_DA_m = np.array([0.40, 0.24, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.24, ])
        self.r_DA_m = r_DA_m / np.max(r_DA_m)

        # Startvation
        if r_dem_a == 'default':
            dem_a = normal(self.a_array, 0, self.a_max)
            r_dem_a = dem_a/np.max(dem_a)*0.3 + 0.7
        self.r_dem_a = r_scale * r_dem_a / np.max(r_dem_a)

        if r_dem_m == 'default':
            r_dem_m = np.array([7, 7.5, 5, 4.5, 4, 3, 2, 3, 4, 4.5, 6, 7])
        self.r_dem_m = r_dem_m / np.mean(r_dem_m)

        if P_alloc == 'default':
            P_alloc = gamma(self.a_array, self.a_max * 0.35, self.a_max * 0.3)
            P_alloc[0] = np.average(P_alloc)
        self.P_alloc = P_alloc / np.sum(P_alloc)

        self.N_duration = N_duration

    def grow(self, steps=1):
        N_adult = np.sum(self.Q * self.adult_prob[:, np.newaxis], axis=0)
        if self.matting_pattern == 'monogamy':
            N_matted = np.min(N_adult)
        elif self.matting_pattern == 'polygamy':
            N_matted = np.min([N_adult[0] * self.N_mmp, N_adult[1]])
        N_breeding = N_matted * self.BR_m[self.date.month-1] / 30 * steps
        self.Q = np.roll(self.Q, steps)
        self.Q[0, :] = [N_breeding / 2, N_breeding / 2]
        return N_breeding

    def hunted(self, E_predators):
        E_predators = np.sum(E_predators)
        E_a_m = self.E_m[self.date.month - 1] * self.E_a[:, np.newaxis]
        R_H = E_predators / np.sum(E_a_m) / 2.0 * np.ones_like(self.Q)
        return -R_H

    def disease_aging(self):
        r_DA = self.r_DA_m[self.date.month - 1] * self.r_DA_a[:, np.newaxis]
        R_DA = r_DA * self.Q
        return -R_DA

    def starvation(self, supply):
        E_demand = self.demand()
        E_supply = np.sum(supply) * self.P_alloc[:, np.newaxis]
        r_sat = E_supply / (E_demand + 1e-12)
        r_sat[r_sat > 1] = 1
        R_ST = self.Q * ((1 - r_sat) ** self.N_duration)
        return -R_ST

    def lack_of_nourish(self, R_reduction):
        r_reduc_adult = np.sum(R_reduction * self.adult_prob[:, np.newaxis]) / \
                        (1e-12 + np.sum(self.Q * self.adult_prob[:, np.newaxis]))
        R_LN = r_reduc_adult * self.Q * self.infant_prob[:, np.newaxis]
        return R_LN

    def step(self, hunter_demand, food_supply, R_Migration=0, Migration_pattern=['senior', 'adults'], steps=1):
        self.grow()

        R_H = self.hunted(hunter_demand)
        R_DA = self.disease_aging()
        R_ST = self.starvation(food_supply)
        RATE = R_H + R_DA + R_ST
        RATE[self.Q + RATE * steps < 0] = -self.Q[self.Q + RATE * steps < 0]
        R_LN = self.lack_of_nourish(RATE)

        if not isinstance(R_Migration, (list, tuple, np.ndarray)):
            prob = 0
            if 'infant' in Migration_pattern:
                prob += self.infant_prob
            if 'juvenile' in Migration_pattern:
                prob += self.juvenile_prob
            if 'adult' in Migration_pattern:
                prob += self.adult_prob
            if 'senior' in Migration_pattern:
                prob += self.adult_prob
            R_Migration = R_Migration / np.sum(self.Q * prob[:, np.newaxis]) * self.Q * prob[:, np.newaxis]
        RATE += R_LN + R_Migration

        RATE[self.Q + RATE * steps < 0] = -self.Q[self.Q + RATE * steps < 0]
        self.Q += RATE * steps
        self.date += datetime.timedelta(steps)


    def energy(self):
        E = self.E_m[self.date.month - 1] * self.E_a[:, np.newaxis] * self.Q
        return E

    def demand(self):
        E = self.energy()
        E_demand = E * self.r_dem_m[self.date.month - 1] * self.r_dem_a[:, np.newaxis]
        return E_demand


class Plants:
    def __init__(self, capacity, tq, th, init=None, init_date=[2022, 5, 1]):
        K = 1 / capacity
        lnb = (np.log(K) - np.log(3 * K)) / (th - tq)
        lna = np.log(K) - lnb * th
        self.a = np.exp(lna)
        self.b = np.exp(lnb)
        self.K = K
        if init == None:
            init = capacity
        self.E = init
        self.date = datetime.date(init_date[0], init_date[1], init_date[2])

    def weight_month(self):
        weight = [0.3000, 0.3001, 0.3031, 0.3378, 0.4694, 0.5793, 0.4695, 0.3702, 0.5425, 0.3325, 0.3000, 0.3001]
        m = self.date.month - 1
        return weight[m]

    def step(self, steps=1):
        K = self.K / self.weight_month()
        a = self.a
        b = self.b
        E = self.E

        if 1 / K > self.E:
            if self.E > 0:
                tau = np.log(1 / a * (1 / E - K)) / np.log(b)
            else:
                tau = 0
            tau += steps
            self.E = 1 / (K + a * b ** tau)
        else:
            self.E = 1 / K
        self.date += datetime.timedelta(steps)
        return self.E