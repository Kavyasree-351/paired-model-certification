"""Does reference-revealed information help? Falsification/replication pilot.

The proposed mechanism is classical two-phase auxiliary-assisted sampling.
This code deliberately distinguishes practical separation from novelty.
No original real-data pool is imported or opened.
"""
import argparse
import hashlib
import json
import time
import unittest
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp
from scipy.stats import beta

ROOT=Path(__file__).resolve().parent
EPS,DELTA=.05,.05
PROTOCOL={'pilot_seeds':[1201,1202,1203],'fresh_seeds':[2201,2202,2203],
 'generator_revision':'v2: homogeneous control retains all three reference categories with identical conditional risk',
 'repetitions_per_seed':20,'label_costs':[1.,20.],'noise_fractions':[.1,.5,1.],
 'true_excess':[.02,.05,.08],'scout_n':120,'max_stream':16000,'cost_cap':30000.,
 'propensity_floor':.05,'confidence':.95,
 'primary':'Post-revelation versus strongest matched precommit certification rate at same total acquisition budget',
 'gate':'learned post improves >=.05 over learned pre in both heterogeneous regimes at costly-label setting; positive in >=2/3 seeds; homogeneous control no manufactured gain; separate novelty gate required',
 'novelty_gate':'Requires theorem or algorithm beyond classical two-phase optimal sampling, AM-PPI and active anytime-valid risk control. Practical improvement alone is insufficient.',
 'final_data':'No original real-data pools loaded; fresh synthetic stage gated on both pilot opportunity and novelty.'}


def parameters(rho,target):
    if rho==1.:return np.repeat(1/3,3),np.repeat(target,3)
    difference=target/.95
    p=np.array([(1-rho+difference)/2,(1-rho-difference)/2,rho])
    if np.min(p)<0:raise ValueError('Impossible target')
    return p,np.array([.95,-.95,0.])


def generate(rng,n,p,m):
    ref=rng.choice(3,n,p=p)
    d=np.where(rng.random(n)<(1+m[ref])/2,1.,-1.)
    # Cheap class 3; reference class in {0,1,2}; observed Y reconstructs D.
    y=np.where(d==1,ref,3)
    return ref,y,d


def scout(rng,p,m,cy):
    ref,y,d=generate(rng,PROTOCOL['scout_n'],p,m)
    counts=np.bincount(ref,minlength=3)
    phat=(counts+.5)/(len(ref)+1.5)
    mhat=np.array([d[ref==k].sum()/(counts[k]+1) for k in range(3)])
    # y is bought for every scout case; all acquisition costs are counted.
    return phat,mhat,float(len(ref)*(1+cy))


def upper(mhat,l):return float(np.max(mhat+(1-mhat)/l))


def variance(p,m,mhat,l):
    mean=float(p@m)
    # AIPW variance remains valid with misspecified fixed mhat.
    second=np.sum(p*(mhat*mhat+2*mhat*(m-mhat)+(1-2*mhat*m+mhat*mhat)/l))
    return float(second-mean*mean)


def expected_growth(p,m,mhat,l,lam):
    # Exact expectation over revealed type, query coin and signed outcome.
    z0=mhat
    zp=mhat+(1-mhat)/l
    zn=mhat+(-1-mhat)/l
    return float(np.sum(p*((1-l)*np.log1p(lam*(EPS-z0))+
       l*(1+m)/2*np.log1p(lam*(EPS-zp))+
       l*(1-m)/2*np.log1p(lam*(EPS-zn)))))


def design(p,m,mhat,cy,post,objective='growth'):
    # Independent pilot/scout predictions only. Same propensity range, optimizer
    # budget and estimator for pre/post; post may condition l on purchased R.
    floor=PROTOCOL['propensity_floor']
    def expand(v):return np.asarray(v[:3]) if post else np.repeat(v[0],3)
    def loss(v):
        l=expand(v);c=1+cy*(p@l)
        if objective=='variance':return c*variance(p,m,mhat,l)
        lam=v[-1]*.95/(upper(mhat,l)-EPS)
        return -expected_growth(p,m,mhat,l,lam)/c
    dim=3 if post else 1
    bounds=[(floor,1.)]*dim+([] if objective=='variance' else [(0.,1.)])
    results=[]
    # Three starts per method, same maximum iterations; no eval-data tuning.
    for base in [.1,.4,.9]:
        init=np.r_[np.full(dim,base),[] if objective=='variance' else [.5]]
        fit=minimize(loss,init,method='L-BFGS-B',bounds=bounds,
                     options={'maxiter':200,'ftol':1e-12})
        results.append(fit)
    best=min(results,key=lambda r:r.fun)
    if not np.isfinite(best.fun):raise RuntimeError('Nonfinite design')
    return expand(best.x),{'objective':objective,'objective_value':float(best.fun),
       'starts':3,'iterations':[int(r.nit) for r in results],
       'optimizer_success':[bool(r.success) for r in results]}


def run_policy(rng,ref,d,mhat,l,cy,scout_cost,raw=False):
    # Label decision uses R only after paying for its purchase. Both comparison
    # methods use the same post-reference mhat; only label timing differs.
    uniforms=rng.random(len(ref))
    label=uniforms<l[ref]
    z=mhat[ref].copy()
    z[label]+=(d[label]-mhat[ref[label]])/l[ref[label]]
    if raw:
        assert np.all(label);z=d.copy()
    spent=scout_cost+np.cumsum(1+cy*label)
    eligible=np.flatnonzero(spent<=PROTOCOL['cost_cap'])
    n=int(eligible[-1]+1) if len(eligible) else 0
    if n==0:return {'certified':False,'cost':scout_cost,'arrivals':0,'labels':0,'z_mean':None}
    bound=1. if raw else upper(mhat,l)
    stakes=np.geomspace(.0005,.95/(bound-EPS),20)
    factors=1+stakes[:,None]*(EPS-z[None,:n])
    assert np.min(factors)>0
    wealth=logsumexp(np.cumsum(np.log(factors),axis=1),axis=0)-np.log(len(stakes))
    passed=np.flatnonzero(wealth>=np.log(1/DELTA))
    stop=int(passed[0]+1) if len(passed) else n
    return {'certified':bool(len(passed)),'cost':float(spent[stop-1]),
       'arrivals':stop,'labels':int(label[:stop].sum()),'z_mean':float(z[:n].mean())}


class Checks(unittest.TestCase):
    def test_reconstructed_loss(self):
        p,m=parameters(.5,.02)
        r,y,d=generate(np.random.default_rng(1),10000,p,m)
        np.testing.assert_array_equal(d,(3!=y).astype(float)-(r!=y).astype(float))

    def test_exact_unbiased_and_null_wealth(self):
        for rho in [.1,.5,1.]:
            p,m=parameters(rho,.05)
            mh=np.array([.7,-.6,.12]);l=np.array([.1,.3,.8])
            ez=np.sum(p*((1-l)*mh+l*(1+m)/2*(mh+(1-mh)/l)+l*(1-m)/2*(mh+(-1-mh)/l)))
            self.assertAlmostEqual(ez,.05,places=12)
            lam=.9/(upper(mh,l)-EPS)
            # One-step e-factor expectation <= 1 under boundary null.
            self.assertLessEqual(1+lam*(EPS-ez),1+1e-12)

    def test_oracle_variance_identity(self):
        p,m=parameters(.5,.02);l=np.array([.1,.2,.8])
        known=np.sum(p*(m-p@m)**2)+np.sum(p*(1-m*m)/l)
        self.assertAlmostEqual(variance(p,m,m,l),known,places=12)

    def test_post_contains_pre(self):
        p,m=parameters(.5,.02)
        a,_=design(p,m,m,20,False,'variance')
        b,_=design(p,m,m,20,True,'variance')
        ca=(1+20*p@a)*variance(p,m,m,a)
        cb=(1+20*p@b)*variance(p,m,m,b)
        self.assertLessEqual(cb,ca+1e-5)

    def test_homogeneous_oracle_allocation(self):
        p,m=parameters(1.,.02)
        a,_=design(p,m,m,20,False,'variance')
        b,_=design(p,m,m,20,True,'variance')
        np.testing.assert_allclose(a,b,atol=1e-4)


def main(stage):
    out=ROOT/'results'/f'revelation_{stage}';out.mkdir(parents=True,exist_ok=True)
    if stage=='fresh':
        gate=json.loads((ROOT/'results/revelation_pilot/gate.json').read_text())
        if not(gate['performance_gate_passed'] and gate['novelty_gate_passed']):
            raise SystemExit('Fresh evaluation refused: opportunity and novelty gates must both pass.')
    protocol=out/'protocol.json'
    if protocol.exists():assert json.loads(protocol.read_text())==PROTOCOL
    else:protocol.write_text(json.dumps(PROTOCOL,indent=2))
    start=time.perf_counter()
    checks=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not checks.wasSuccessful():raise SystemExit(1)
    rows=[];designs=[]
    for rho in PROTOCOL['noise_fractions']:
        for cy in PROTOCOL['label_costs']:
            for target in PROTOCOL['true_excess']:
                p,m=parameters(rho,target)
                for seed in PROTOCOL[stage+'_seeds']:
                    rng=np.random.default_rng(seed+int(rho*1000)+int(cy*100)+int(target*10000))
                    ph,mh,sc=scout(rng,p,m,cy)
                    policies={'raw_all':(np.ones(3),np.zeros(3),0.,True)}
                    for prefix,pp,mm,mmhat,scost in [('learned',ph,mh,mh,sc),('oracle',p,m,m,0.)]:
                        for timing in ['pre','post']:
                            l,meta=design(pp,mm,mmhat,cy,timing=='post')
                            method=f'{prefix}_{timing}'
                            policies[method]=(l,mmhat,scost,False)
                            designs.append({'rho':rho,'label_cost':cy,'target':target,'seed':seed,'method':method,
                              'l':l.tolist(),'mhat':mmhat.tolist(),'scout_cost':scost,
                              'true_variance_times_cost':(1+cy*(p@l))*variance(p,m,mmhat,l),**meta})
                    for rep in range(PROTOCOL['repetitions_per_seed']):
                        ref,y,d=generate(rng,PROTOCOL['max_stream'],p,m)
                        for method,(l,mmhat,scost,raw) in policies.items():
                            # Common label coins across all methods, matched cases.
                            draw=np.random.default_rng(seed+rep*100000)
                            value=run_policy(draw,ref,d,mmhat,l,cy,scost,raw)
                            rows.append({'rho':rho,'label_cost':cy,'target':target,'seed':seed,'rep':rep,
                               'method':method,'scout_cost':scost,**value})
                print('complete',rho,cy,target,flush=True)
    summary=[]
    for rho in PROTOCOL['noise_fractions']:
        for cy in PROTOCOL['label_costs']:
            for target in PROTOCOL['true_excess']:
                for method in sorted(set(r['method'] for r in rows)):
                    group=[r for r in rows if (r['rho'],r['label_cost'],r['target'],r['method'])==(rho,cy,target,method)]
                    k=sum(r['certified'] for r in group);n=len(group)
                    summary.append({'rho':rho,'label_cost':cy,'target':target,'method':method,'n':n,'rate':k/n,
                      'ci95':[0 if k==0 else float(beta.ppf(.025,k,n-k+1)),1 if k==n else float(beta.ppf(.975,k+1,n-k))],
                      'mean_cost':float(np.mean([r['cost'] for r in group]))})
    gains=[]
    for rho in [.1,.5,1.]:
        pick=lambda method:next(r for r in summary if r['rho']==rho and r['label_cost']==20 and r['target']==.02 and r['method']==method)
        differences=[]
        for seed in PROTOCOL[stage+'_seeds']:
            rates=[]
            for method in ['learned_post','learned_pre']:
                g=[r for r in rows if r['rho']==rho and r['label_cost']==20 and r['target']==.02 and r['seed']==seed and r['method']==method]
                rates.append(np.mean([r['certified'] for r in g]))
            differences.append(float(rates[0]-rates[1]))
        gains.append({'rho':rho,'gain':pick('learned_post')['rate']-pick('learned_pre')['rate'],
                      'seed_gains':differences,'positive_seed_fraction':float(np.mean(np.array(differences)>0))})
    opportunity=all(r['gain']>=.05 and r['positive_seed_fraction']>=2/3 for r in gains[:2])
    gate={'performance_gate_passed':opportunity,'novelty_gate_passed':False,'gains':gains,
      'reason':'Post-reference label allocation is established two-phase sampling; finite-sample betting wrapper is known. No distinct theorem beyond these ingredients established.',
      'decision':'No fresh/full evaluation or complete novel-method paper.'}
    for name,value in [('raw',rows),('designs',designs),('summary',summary),('gate',gate)]:
        (out/f'{name}.json').write_text(json.dumps(value,indent=2))
    (out/'execution_metadata.json').write_text(json.dumps({'seconds_excluding_imports':time.perf_counter()-start,
        'checks_passed':checks.testsRun,'policy_evaluations':len(rows),'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest()},indent=2))
    print(json.dumps(gate,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['pilot','fresh'],default='pilot')
    main(ap.parse_args().stage)
