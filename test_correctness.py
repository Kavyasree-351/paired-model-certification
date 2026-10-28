import unittest
import numpy as np
from scipy.special import gammaln
from certificates import certified, choose, universal_log_e, METHODS
from research import partitions


class Correctness(unittest.TestCase):
    def test_partitions(self):
        parts=partitions(1000,11)
        self.assertEqual(len(set(np.concatenate(parts))),1000)
        self.assertEqual([len(x) for x in parts],[400,100,200,100,100,100])
        other=partitions(1000,22)
        for a,b in zip(parts[2:],other[2:]):
            np.testing.assert_array_equal(a,b)

    def test_extremes(self):
        for method in METHODS:
            self.assertTrue(certified(np.zeros(10000),.05,.05,method),method)
            self.assertFalse(certified(np.ones(10000),.05,.05,method),method)

    def test_no_restart(self):
        # First candidate harmful, second safe: fixed sequence MUST stop.
        d=np.column_stack([np.ones(10000),np.zeros(10000)])
        self.assertEqual(choose(d,.05,.05,'harm_only'),-1)
        self.assertEqual(choose(d,.05,.05,'harm_only','bonferroni'),1)

    def test_permutation_invariance(self):
        d=np.array([-1]*30+[0]*900+[1]*70)
        self.assertAlmostEqual(universal_log_e(d,.05),universal_log_e(d[::-1],.05),places=10)

    def test_null_monte_carlo(self):
        # Empirical size check at boundary, not a proof; 400 repetitions.
        rng=np.random.default_rng(992)
        for method in METHODS[:-1]:
            rejections=sum(certified(rng.choice([-1,0,1],500,p=[.1,.75,.15]),.05,.05,method) for _ in range(400))
            self.assertLessEqual(rejections,35,(method,rejections))

    def test_e_value_exact_small_sample(self):
        # Enumerate every count vector, independently weighting ordered sequences.
        n=6
        for probs in [[.05,.85,.10],[.3,.35,.35],[0.,.95,.05]]:
            for fitted in [None,np.array([.02,.95,.03])]:
                expectation=0.
                for a in range(n+1):
                    for h in range(n-a+1):
                        z=n-a-h
                        multiplicity=np.exp(gammaln(n+1)-sum(gammaln(k+1) for k in [a,z,h]))
                        probability=multiplicity*np.prod([p**k for p,k in zip(probs,[a,z,h])])
                        d=np.array([-1.]*a+[0.]*z+[1.]*h)
                        expectation+=probability*np.exp(universal_log_e(d,.05,fitted))
                self.assertLessEqual(expectation,1+1e-7)


if __name__=='__main__':unittest.main()
