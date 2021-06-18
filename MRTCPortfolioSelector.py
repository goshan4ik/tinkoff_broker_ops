from online_portfolio_selection_general import OnlinePortfolioSelector
import numpy as np
from scipy.optimize import bisect
from numpy.linalg import norm


def g(c, rate, prev_adj_portfolio, portfolio):
    return func(c, rate, prev_adj_portfolio, portfolio)


def func(c, rate, prev_adj_portfolio, portfolio):
    return c + rate * norm(prev_adj_portfolio - np.multiply(portfolio, c)) - 1


class MRTCPortfolioSelector(OnlinePortfolioSelector):

    def __init__(self, stocks_count: int, rate: float, weight: float):
        super().__init__()
        self.__stocks_count = stocks_count
        self.__cumulative_return = 1
        self.__rate = rate
        self.__weight = weight
        self.__price_relations_history = []
        self.__portfolio_history = [np.full(stocks_count, 1/stocks_count)]
        self.__adjusted_portfolio_history = [np.zeros(stocks_count)]
        self.memory = {x: 1 for x in range(0, stocks_count)}

    def on_price_relations_updated(self, price_relations: np.array):
        self.__price_relations_history.append(price_relations)
        c = self.__calc_c()
        prev_wealth_zero_tc = self.__calc_period_wealth_with_zero_transaction_cost(self.__portfolio_history[-1],
                                                                                   price_relations)
        self.__cumulative_return = self.__cumulative_return * c * prev_wealth_zero_tc
        closing_adjusted_portfolio = self.__calc_closing_adjusted_portfolio(price_relations, prev_wealth_zero_tc)
        self.__adjusted_portfolio_history.append(closing_adjusted_portfolio)
        transfers_count = self.__calc_transfers_count()
        period = len(self.__portfolio_history)
        print(period)
        next_period_portfolio = self.__select_portfolio(transfers_count, closing_adjusted_portfolio)
        self.__portfolio_history.append(next_period_portfolio)
        if period % 50 == 0:
            print(self.__cumulative_return)

    def __calc_c(self) -> float:
        adj_portfolio = self.__adjusted_portfolio_history[-1]
        p = self.__portfolio_history[0]
        return bisect(g, -1, 1, args=(self.__rate, adj_portfolio, p))

    def __calc_period_wealth_with_zero_transaction_cost(self, portfolio, price_relations: np.array) -> float:
        return np.dot(portfolio, price_relations)

    def __calc_closing_adjusted_portfolio(self, price_relations: [float], current_period_wealth) -> [float]:
        current_portfolio = self.__portfolio_history[-1]
        closing_adjusted_portfolio = np.zeros(self.__stocks_count)
        for i in range(0, self.__stocks_count):
            closing_adjusted_portfolio[i] = (current_portfolio[i] * price_relations[i]) / current_period_wealth
        return closing_adjusted_portfolio

    def __calc_current_period_wealth(self, price_relations: [float]) -> float:
        wealth = 0
        current_portfolio = self.__portfolio_history[-1]
        for i in range(0, self.__stocks_count):
            wealth = wealth + current_portfolio[i] * price_relations[i]
        return wealth

    # def __calc_transfers_count(self) -> int:
    #     best_transfers_count = 0
    #     best_cumulative_wealth = -float("inf")
    #     for transfers_count in range(0, self.__stocks_count):
    #         cumulative_wealth = 1
    #         for t in range(0, len(self.__portfolio_history)):
    #             portfolio = self.__select_portfolio(t, transfers_count, self.__adjusted_portfolio_history[t])
    #             cumulative_wealth = cumulative_wealth * self.__calc_period_wealth_with_zero_transaction_cost(
    #                 self.__price_relations_history[t], portfolio)
    #         if cumulative_wealth > best_cumulative_wealth:
    #             best_cumulative_wealth = cumulative_wealth
    #             best_transfers_count = transfers_count
    #     return best_transfers_count

    def __calc_transfers_count(self) -> int:
        best_transfers_count = 0
        best_cumulative_wealth = -float("inf")
        for transfers_count in range(0, self.__stocks_count):
            cumulative_wealth = self.memory[transfers_count]
            portfolio = self.__select_portfolio(transfers_count, self.__adjusted_portfolio_history[-1])
            cumulative_wealth = cumulative_wealth * self.__calc_period_wealth_with_zero_transaction_cost(
                            self.__price_relations_history[-1], portfolio)
            if cumulative_wealth > best_cumulative_wealth:
                best_cumulative_wealth = cumulative_wealth
                best_transfers_count = transfers_count
        return best_transfers_count

    def __select_portfolio(self, transfers_count, closing_adjusted_portfolio) -> np.array:
        portfolio = np.zeros(closing_adjusted_portfolio.size)
        x = np.zeros(self.__stocks_count)
        for i in range(0, self.__stocks_count):
            x[i] = self.__metric_1(i, self.__weight)
        indices = x.argsort()[::-1]
        portfolio[-1] = closing_adjusted_portfolio[indices[-1]]
        for i in range(0, transfers_count):
            portfolio[i] = 0
            portfolio[-1] += closing_adjusted_portfolio[indices[i]]
        for i in range(transfers_count, self.__stocks_count - 1):
            portfolio[i] = closing_adjusted_portfolio[indices[i]]

        return portfolio

# 1.0579216282212873
    # 1.129382304872042
    # 1.2487438779846216
    # 1.283103484053459

    #  TODO fix
    def __metric_1(self, i, a):
        t = len(self.__price_relations_history)
        x = self.__price_relations_history
        value = 1
        for k in range(0, t):
            value *= x[k][i]**((1 - a) ** (t - k) * a)
        return value

    #  TODO fix
    def __metric_2(self, t, i, a):
        if t == 1:
            return 1
        x = self.__price_relations_history
        if t == 2:
            return a * x[1][i] + (1 - a) * x[1][i]
        return a * x[t - 1][i] + (1 - a) * self.__metric_2(t-2, i, a) * x[t - 1][i]
