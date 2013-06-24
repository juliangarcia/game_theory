'''
Created on Jan 31, 2013

@author: garcia
'''
import unittest
from game_theory import BimatrixTwoStrategyGame
from game_theory import NoEquilibriumSelected
import numpy.random as random


class Test(unittest.TestCase):

    def testUniqueEquilibrium(self):
        game_with_unique_equilibrium = BimatrixTwoStrategyGame(
            a1=1, a2=4, b1=3, b2=3, c1=2, c2=2, d1=4, d2=1)
        self.assertEqual(1, len(game_with_unique_equilibrium.find_nash()))

    def testSymmetryChecker(self):
        symmetric_game = BimatrixTwoStrategyGame.fromsymmetricpayoffs(
            random.rand(), random.rand(), random.rand(), random.rand())
        self.assertTrue(symmetric_game.is_symmetric())

    def testEquilibriumSelectionPD(self):
        a_pd = BimatrixTwoStrategyGame(
            -1.0, -1.0, -4.0, 0.0, 0.0, -4.0, -3.0, -3.0)
        # equilibrium is unique
        self.assertEqual(len(a_pd.find_nash()), 1)
        # D,D
        self.assertTrue(a_pd.find_unique_equilibrium() == (0.0, 0.0))
        # no risk dominant
        try:
            a_pd.find_risk_dominant_equilibrium(atol=10e-3)
        except NoEquilibriumSelected:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def testEquilibriumSelectionCoordinationGame(self):
        a_coordination_game = BimatrixTwoStrategyGame(
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
        self.assertEqual(len(a_coordination_game.find_nash()), 3)
        self.assertEqual(
            a_coordination_game.find_unique_equilibrium(), (0.5, 0.5))

    def testEquilibriumSelectionBattleOfTheSexes(self):
        battle = BimatrixTwoStrategyGame.battleofthesexes()
        self.assertEqual(len(battle.find_nash()), 3)
        try:
            battle.find_unique_equilibrium()
        except NoEquilibriumSelected:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def testEquilibriumSelectionMatchingPennies(self):
        pennies = BimatrixTwoStrategyGame.matchingpennies()
        self.assertEqual(len(pennies.find_nash()), 1)
        self.assertEqual(pennies.find_nash()[0], (0.5, 0.5))

    def testRiskDominanceOffDiagonal(self):
        test_game = BimatrixTwoStrategyGame(6.0,6.0,4.9,7.0,6.1,4.0,3.0,3.0)
        self.assertEqual(test_game.find_risk_dominant_equilibrium(), (1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
