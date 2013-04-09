'''
    Created on Mar 23, 2012

    @author: garcia
'''
import numpy as np
import matplotlib.pyplot as plt


class NoEquilibriumSelected(Exception):
    """
    An exception to be thrown when a given function cannot select a unique equilibrium.
    """
    def __innit__(self, errno, msg):
        self.args = (errno, msg)


class BimatrixTwoStrategyGame:
    """
    Holds a bimatrix game data,
    and finds equilibria.
    """
    def __init__(self, a1, a2, b1, b2, c1, c2, d1, d2):
        self.a1 = float(a1)
        self.b1 = float(b1)
        self.c1 = float(c1)
        self.d1 = float(d1)
        self.a2 = float(a2)
        self.b2 = float(b2)
        self.c2 = float(c2)
        self.d2 = float(d2)

    @classmethod
    def fromsymmetricpayoffs(cls, a1, b1, c1, d1):
        """
        Create a symmetric game
        """
        symmetric_game = cls(a1, a1, b1, c1, c1, b1, d1, d1)
        return symmetric_game

    @classmethod
    def prisonersdilemma(cls, reward=3.0, sucker=0.0, temptation=4.0, punishment=1.0):
        """
        Create a prisoners dilemma
        """
        a1 = reward
        b1 = sucker
        c1 = temptation
        d1 = punishment
        pd = cls(a1, a1, b1, c1, c1, b1, d1, d1)
        return pd

    @classmethod
    def battleofthesexes(cls):
        """
        Create a battle of the sexes game (see pg 11 on Gibbons book).
        """
        return cls(2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0)

    @classmethod
    def matchingpennies(cls):
        """
        Create a matching pennies game (see pg 29 on Gibbons book).
        """
        return cls(-1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0)

    def __str__(self):
        return '[(' + str(self.a1) + ',' + str(self.a2) + '), (' + str(self.b1) + ',' + str(self.b2) + '), \n (' + str(self.c1) + ',' + str(self.c2) + '), (' + str(self.d1) + ',' + str(self.d2) + ')]'

    def find_pure_nash(self):
        """
        Finds pure equilibria by exahustively testing the four possibilities.
        Returns a list of tuples (p,q) where p is the probability of player 1 playing the first strategy,
        and q is the probability of player 2 playing the first strategy.
        """
        ans = []
        if (self.a1 >= self.c1 and self.a2 >= self.b2):
            ans.append((1.0, 1.0))
            #test p=1,q=0
        if (self.b1 >= self.d1 and self.b2 >= self.a2):
            ans.append((1.0, 0.0))
            #test p=0,q=1.0
        if (self.c1 >= self.a1 and self.c2 >= self.d2):
            ans.append((0.0, 1.0))
        #test p=0,q=0.0
        if (self.d1 >= self.b1 and self.d2 >= self.c2):
            ans.append((0.0, 0.0))
        return ans

    def find_mixed_nash(self):
        """
        Finds the mixed equilibria, if it exists.
        """
        ans = []
        try:
            q = (self.d1 - self.b1) / (self.a1 - self.b1 - self.c1 + self.d1)
            p = (self.d2 - self.c2) / (self.a2 - self.b2 - self.c2 + self.d2)
        except ZeroDivisionError:
            return ans
        if (0 < p < 1 and 0 < q < 1):
            ans.append((p, q))
        return ans

    def find_nash(self):
        """
        Returns the set of Nash equilibria.
        """
        return self.find_pure_nash() + self.find_mixed_nash()

    def find_risk_dominant_equilibrium(self, atol=10e-3):
        """
        Attemps to select one equilibrium by risk dominance.
        """
        candidates = self.find_nash()
        if (len(candidates) == 1):
            return candidates[0]
        if (len(candidates) > 3):
            raise ValueError("Degenerate case")
        if ((0.0, 1.0) in candidates):
            #we are off diagonal
            a = (self.d1 - self.b1) * (self.a2 - self.b2)
            b = (self.a1 - self.c1) * (self.d2 - self.c2)
            if (a == b):
                raise NoEquilibriumSelected("No risk dominant equilibrium for game: " + str(self))
            elif (a > b):
                toBeFlipped = (0.0, 0.0)
            elif (a < b):
                toBeFlipped = (1.0, 1.0)
            return (1.0 - toBeFlipped[0], toBeFlipped[1]), "Risk dominance"
        else:
            #we are on diagonal
            a = (self.b1 - self.d1) * (self.c2 - self.d2)
            b = (self.c1 - self.a1) * (self.b2 - self.a2)
            if (a == b):
                raise NoEquilibriumSelected("No risk dominant equilibrium for game: " + str(self))
            elif (a > b):
                return (0.0, 0.0)
            elif (a < b):
                return (1.0, 1.0)
        raise NoEquilibriumSelected("No risk dominant equilibrium for game: " + str(self))

    def find_payoff_dominant_equilibrium(self, atol=10e-3):
        """
        Attemps to select one equilibrium  by payoff dominance.
        """
        candidates = self.find_nash()
        if (len(candidates) == 1):
            return candidates[0]
        if (len(candidates) > 3):
            raise ValueError("Degenerate case")
        if ((0.0, 1.0) in candidates):
            #we are off diagonal
            if ((self.c1 >= self.b1 and self.c2 >= self.b2) and (self.c1 > self.b1 or self.c2 > self.b2)):
                return (0.0, 1.0)
            if ((self.c1 <= self.b1 and self.c2 <= self.b2) and (self.c1 < self.b1 or self.c2 < self.b2)):
                return (1.0, 0.0)
            raise NoEquilibriumSelected("No payoff dominant equilibrium for game: " + str(self))
        else:
            #we are on diagonal
            if ((self.a1 >= self.d1 and self.a2 >= self.d2) and (self.a1 > self.d1 or self.a2 > self.d2)):
                return (1.0, 1.0)
            if ((self.a1 <= self.d1 and self.a2 <= self.d2) and (self.a1 < self.d1 or self.a2 < self.d2)):
                return (0.0, 0.0)
            raise NoEquilibriumSelected("No payoff dominant equilibrium for game: " + str(self))

    def find_focal_equilibrium_by_symmetry(self, atol=10e-3):
        """
        If it exists, returns a mixed equilibrium if it is also symmetric
        """
        candidates = self.find_nash()
        #attemp focal symmetry
        for profile in candidates:
            if(np.abs(profile[0] - profile[1]) < atol):
                if (profile[0] < 1.0 or profile[0] > 0.0):
                    return profile
        raise NoEquilibriumSelected("No focal symmetry for game: " + str(self))

    def find__one_pareto_optimal_equilibrium(self, atol=10e-3):
        candidates = self.find_nash()
        return candidates[np.argmax([sum(self.expected_payoff(candidate)) for candidate in candidates])]

    def expected_payoff(self, profile):
        """
        Computes the expected payoff given a profile
        """
        p = profile[0]
        q = profile[1]
        pi_1 = p * q * self.a1 + p * (1.0 - q) * self.b1 + (
            1.0 - p) * q * self.c1 + (1.0 - p) * (1.0 - q) * self.d1
        pi_2 = p * q * self.a2 + p * (1.0 - q) * self.b2 + (
            1.0 - p) * q * self.c2 + (1.0 - p) * (1.0 - q) * self.d2
        return (pi_1, pi_2)

    def plot_payoff_space(self, size_x=5, size_y=5, grid_on=True, select_equilibrium=True):
        fig = plt.figure(figsize=(size_x, size_y))
        maximum_payoff = max(self.a1, self.a2, self.b1, self.b2,
                             self.c1, self.c2, self.d1, self.d2)
        minimum_payoff = min(self.a1, self.a2, self.b1, self.b2,
                             self.c1, self.c2, self.d1, self.d2)
        plt.xlim(minimum_payoff - 1, maximum_payoff + 1)
        plt.ylim(minimum_payoff - 1, maximum_payoff + 1)
        #first arrow connecting (a1,a2) with (c1,c2)
        if self.a1 < self.c1:
            plt.arrow(self.a1, self.a2, self.c1 - self.a1, self.c2 - self.a2, head_width=.15, length_includes_head=True, facecolor='black')
        elif self.a1 > self.c1:
            plt.arrow(self.c1, self.c2, self.a1 - self.c1, self.a2 - self.c2, head_width=.15, length_includes_head=True, facecolor='black')
        else:
            plt.arrow(self.a1, self.a2, self.c1 - self.a1, self.c2 - self.a2, head_width=0, length_includes_head=True, facecolor='black')
        #second arrow connecting (b1,b2) with (d1,d2)
        if self.b1 < self.d1:
            plt.arrow(self.b1, self.b2, self.d1 - self.b1, self.d2 - self.b2, head_width=.15, length_includes_head=True, facecolor='black')
        elif self.b1 > self.d1:
            plt.arrow(self.d1, self.d2, self.b1 - self.d1, self.b2 - self.d2, head_width=.15, length_includes_head=True, facecolor='black')
        else:
            plt.arrow(self.b1, self.b2, self.d1 - self.b1, self.d2 - self.b2, head_width=0, length_includes_head=True, facecolor='black')
        #third arrow connecting (a1,a2) with (b1,b2)
        if self.a2 < self.b2:
            plt.arrow(self.a1, self.a2, self.b1 - self.a1, self.b2 - self.a2, head_width=.15, length_includes_head=True, linestyle='dashed', facecolor='green', edgecolor='green')
        elif self.a2 > self.b2:
            plt.arrow(self.b1, self.b2, self.a1 - self.b1, self.a2 - self.b2, head_width=.15, length_includes_head=True, linestyle='dashed', facecolor='green', edgecolor='green')
        else:
            plt.arrow(self.a1, self.a2, self.b1 - self.a1, self.b2 - self.a2, head_width=0, length_includes_head=True, linestyle='dashed', facecolor='green', edgecolor='green')
        #fourth arrow connecting (c1,c2) with (d1,d2)
        if self.c2 < self.d2:
            plt.arrow(self.c1, self.c2, self.d1 - self.c1, self.d2 - self.c2, head_width=.15, length_includes_head=True, linestyle='dashed', facecolor='green', edgecolor='green')
        elif self.c2 > self.d2:
            plt.arrow(self.d1, self.d2, self.c1 - self.d1, self.c2 - self.d2, head_width=.15, length_includes_head=True, linestyle='dashed', facecolor='green', edgecolor='green')
        else:
            plt.arrow(self.c1, self.c2, self.d1 - self.c1, self.d2 - self.c2, head_width=0, length_includes_head=True, linestyle='dashed', facecolor='green', edgecolor='green')
        #now plot nash
        profiles = self.find_nash()
        for profile in profiles:
            (pi_1, pi_2) = self.expected_payoff(profile)
            plt.plot(pi_1, pi_2, 'bo-')
        if select_equilibrium and len(profiles) > 1:
            try:
                selected = self.find_risk_dominant_equilibrium()
                (pi_1, pi_2) = self.expected_payoff(selected)
                plt.plot(pi_1, pi_2, 'ro-', label='Risk dominant')
            except NoEquilibriumSelected:
                pass
            try:
                selected = self.find_payoff_dominant_equilibrium()
                (pi_1, pi_2) = self.expected_payoff(selected)
                plt.plot(pi_1, pi_2, 'go-', label='Payoff dominant')
            except NoEquilibriumSelected:
                pass
        plt.xlabel('Row player')
        plt.ylabel('Column player')
        plt.grid(grid_on)
        plt.close()
        return fig
