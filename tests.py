'''
Created on Jan 31, 2013

@author: garcia
'''
import unittest
from game_theory import BimatrixTwoStrategyGame
import numpy.random as random

class Test(unittest.TestCase):

    def testUniqueEquilibrium(self):
        game_with_unique_equilibrium = BimatrixTwoStrategyGame(a1=1, a2=4, b1=3, b2=3, c1=2, c2=2, d1=4, d2=1)
        self.assertEqual(1, len(game_with_unique_equilibrium.find_nash()))

    def testSymmetryChecker(self):
        symmetric_game = BimatrixTwoStrategyGame.fromsymmetricpayoffs(random.rand(),random.rand(),random.rand(),random.rand())
        self.assertTrue(symmetric_game.is_symmetric())

if __name__ == "__main__":
    unittest.main()
