"""Smoke test of TWO-STAGE calibration acquisition, not a novel algorithm.

No existing real-data evaluation pools are accessed. Fixed synthetic policies,
IID observations, known query probabilities, and ordinary importance weighting.
All simulations retain every predefined seed. This is not a powered pilot.
"""
import json
import time
import unittest
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta

EPS,DELTA=.05,.05
ROOT=Path(__file__).resolve().parent


def generate(rng,n,target):
    high=rng.random(n)<.3
    disagreement=rng.random(n)<np.where(high,.5,.04)
    low_sign=-.02
    high_sign=(target-.7*.04*low_sign)/(.3*.5)
    assert abs(high_sign)<=1
    harmful=rng.random(n)<(1+np.where(high,high_sign,low_sign))/2
    cheap=disagreement.astype(int)
    reference=np.zeros(n,dtype=int)
    labels=np.where(disagreement,~harmful,rng.random(n)<.1).astype(int)
    d=(cheap!=labels).astype(float)-(reference!=labels).astype(float)
    return high,cheap,reference,labels,d


def acquire(rng,high,cheap,reference,labels,method):
    n=len(high)
    if method=='all':r=np.ones(n)
    elif method=='agreement_skip':r=np.ones(n)
    elif method=='joint_fixed':r=np.where(high,.9,.25)
    elif method=='uniform_reference':r=np.full(n,.3*.9+.7*.25)
    else:raise ValueError(method)
    got_ref=rng.random(n)<r
    # Reference predictions are observed ONLY at got_ref rows.
    disagree=np.zeros(n,dtype=bool)
    disagree[got_ref]=cheap[got_ref]!=reference[got_ref]
    if method=='all':l=np.ones(n)
    elif method in ['agreement_skip','uniform_reference']:l=np.ones(n)
    else:l=np.where(high,1.,.5)
    got_label=got_ref & (rng.random(n)<l)
    if method!='all':got_label &= disagree
    z=np.zeros(n)
    observed=got_label
    # Labels are indexed only AFTER both acquisition decisions.
    d=(cheap[observed]!=labels[observed]).astype(float)-(reference[observed]!=labels[observed]).astype(float)
    z[observed]=d/(r[observed]*l[observed])
    # Zero label propensity is valid on observed agreement: D is known zero.
    bound=1. if method in ['all','agreement_skip'] else (8. if method=='joint_fixed' else 1/(.3*.9+.7*.25))
    return z,got_ref,got_label,disagree,bound


def crossing(z,bound):
    # Nonnegative wealth under H0 E[D]>=eps; fixed predictable acquisition.
    stakes=np.geomspace(.002,.95/(bound-EPS),24)
    factors=1+stakes[:,None]*(EPS-z[None,:])
    if np.min(factors)<=0:raise AssertionError('Bankruptcy: incorrect bound')
    wealth=logsumexp(np.cumsum(np.log(factors),axis=1),axis=0)-np.log(len(stakes))
    passed=np.flatnonzero(wealth>=np.log(1/DELTA))
    return int(passed[0]+1) if len(passed) else None


class Smoke(unittest.TestCase):
    def test_agreement_identity(self):
        rng=np.random.default_rng(1)
        high,c,r,y,d=generate(rng,10000,.02)
        self.assertTrue(np.all(d[c==r]==0))

    def test_skip_preserves_entire_loss_sequence(self):
        high,c,r,y,d=generate(np.random.default_rng(2),10000,.02)
        z,qr,ql,dis,b=acquire(np.random.default_rng(3),high,c,r,y,'agreement_skip')
        np.testing.assert_array_equal(z,d)
        self.assertEqual(int(ql.sum()),int((c!=r).sum()))
        self.assertEqual(crossing(z,b),crossing(d,1.))

    def test_two_stage_bound_and_observation(self):
        high,c,r,y,d=generate(np.random.default_rng(4),10000,.02)
        z,qr,ql,dis,b=acquire(np.random.default_rng(5),high,c,r,y,'joint_fixed')
        self.assertTrue(np.all(~ql | qr))
        self.assertTrue(np.all(~ql | dis))
        self.assertLessEqual(np.max(abs(z)),b)
        self.assertTrue(np.all(z[~ql]==0))

    def test_ipw_unbiased_with_fixed_cases(self):
        high,c,r,y,d=generate(np.random.default_rng(6),2000,.02)
        estimates=[]
        rng=np.random.default_rng(7)
        for _ in range(500):
            z,*_=acquire(rng,high,c,r,y,'joint_fixed');estimates.append(z.mean())
        se=np.std(estimates,ddof=1)/np.sqrt(len(estimates))
        self.assertLess(abs(np.mean(estimates)-d.mean()),4*se+.0001)


def run():
    start=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Smoke))
    if not result.wasSuccessful():raise SystemExit(1)
    rows=[]
    methods=['all','agreement_skip','uniform_reference','joint_fixed']
    for target in [.02,.05,.08]:
        for seed in [71,72,73]:
            rng=np.random.default_rng(seed+int(target*10000))
            for repetition in range(100):
                high,c,r,y,d=generate(rng,3000,target)
                for method in methods:
                    z,qr,ql,dis,b=acquire(rng,high,c,r,y,method)
                    stop=crossing(z,b)
                    horizon=stop or len(z)
                    rows.append({'true_excess':target,'seed':seed,'rep':repetition,'method':method,
                      'certified':stop is not None,'samples_to_certificate':stop,
                      'reference_calls':int(qr[:horizon].sum()),'labels':int(ql[:horizon].sum()),
                      'mean_ipw_full_horizon':float(z.mean()),'mean_complete_full_horizon':float(d.mean())})
        print('completed true excess',target,flush=True)
    summary=[]
    for target in [.02,.05,.08]:
        for method in methods:
            group=[r for r in rows if r['true_excess']==target and r['method']==method]
            k=sum(r['certified'] for r in group);n=len(group)
            summary.append({'true_excess':target,'method':method,'trials':n,'certifications':k,
              'certification_rate':k/n,
              'rate_ci95':[0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))],
              'mean_reference_calls_until_stop_or_cap':float(np.mean([r['reference_calls'] for r in group])),
              'mean_labels_until_stop_or_cap':float(np.mean([r['labels'] for r in group])),
              'mean_ipw_bias':float(np.mean([r['mean_ipw_full_horizon']-r['mean_complete_full_horizon'] for r in group]))})
    elapsed=time.perf_counter()-start
    payload={'tests_passed':result.testsRun,'seconds_excluding_imports':elapsed,
      'n_per_trial':3000,'seeds':[71,72,73],'repetitions_per_seed':100,'summary':summary,
      'scope':'Known IPW and betting machinery, fixed synthetic acquisition policy; no learned allocation, optimality result or novelty claim. No real evaluation/test pools used.',
      'interpretation':'Certification rates must be compared together with resource costs. Costs include failed attempts up to cap; lower spend alone is not an improvement.'}
    out=ROOT/'results'/'selective_smoke';out.mkdir(parents=True,exist_ok=True)
    (out/'raw.json').write_text(json.dumps(rows,indent=2))
    (out/'summary.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2))


if __name__=='__main__':run()
