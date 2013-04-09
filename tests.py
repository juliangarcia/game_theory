'''
Created on Jan 31, 2013

@author: garcia
'''
import unittest
from game_theory import BimatrixTwoStrategyGame


class Test(unittest.TestCase):

    def testFixation(self):
        game_with_unique_equilibrium = BimatrixTwoStrategyGame(a1=1, a2=4, b1=3, b2=3, c1=2, c2=2, d1=4, d2=1)
        self.assertEqual(1, len(game_with_unique_equilibrium.find_nash()))

if __name__ == "__main__":
    unittest.main()
