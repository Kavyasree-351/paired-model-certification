"""Falsification pilot for staged acquisition; no final real data accessed.

Key restricted-class result: when label probability l(X) depends only on
pre-query X (apart from skipping agreement), (q,l) is dominated by (q*l,1).
The observed signed-IPW loss distribution is identical, labels cost the same,
and reference calls decrease. This is an elementary result, not new theory.
"""
import json
import time
import unittest
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,brentq
from scipy.special import logsumexp
from scipy.stats import beta
from selective_smoke import generate,acquire

ROOT=Path(__file__).resolve().parent
W=np.array([.7,.3]);G=np.array([.04,.5]);EPS=.05;DELTA=.05
PROTOCOL={'seeds':[811,822,833],'reps_per_seed':50,'n_scout':500,'max_stream':10000,
 'generator':'binary paired losses with hidden common random label permutation',
 'reference_cost':1.,'label_costs':[.2,5.],'total_cost_cap':4000.,
 'target_excess':[.02,.05,.08],
 'primary':'safe-target certification success at matched total acquisition cost',
 'gate':'learned joint allocation improves >= .05 over strongest feasible comparator in both cost settings; positive gains in >=2/3 seeds; no null alarm > .05; genuinely distinct technical claim required',
 'scope':'Oracle/learned context-only probabilities; scout cost included; not a reproduction of AM-PPI; real held-out pools untouched.'}


def hidden_data(rng,n,target):
    high,c,r,y,d=generate(rng,n,target)
    # A common random permutation preserves paired loss but prevents the cheap
    # prediction itself from revealing disagreement with a constant reference.
    flip=rng.integers(0,2,n)
    return high,c^flip,r^flip,y^flip,d


def moment(q,l,g=G):return float(np.sum(W*g/(q*l)))


def cost(q,l,cy,g=G):return float(np.sum(W*q*(1+cy*g*l)))


def collapsed_opt(g,cy,budget):
    # Min E[g/q] at a fixed per-arrival expected acquisition cost.
    c=1+cy*g
    def q(a):return np.minimum(1,np.sqrt(np.maximum(g,1e-10)/(a*c)))
    if np.sum(W*c)<=budget:return np.ones(2)
    a=brentq(lambda a:np.sum(W*c*q(a))-budget,1e-12,1e8)
    return q(a)


def nested_opt(g,cy,budget):
    fun=lambda v:moment(v[:2],v[2:],g)
    cons={'type':'ineq','fun':lambda v:budget-cost(v[:2],v[2:],cy,g)}
    q=collapsed_opt(g,cy,budget)
    result=minimize(fun,np.r_[q,[1.,1.]],method='SLSQP',bounds=[(.005,1.)]*4,
      constraints=cons,options={'ftol':1e-10,'maxiter':500})
    if not result.success:raise RuntimeError(result.message)
    return result.x[:2],result.x[2:]


class AuditTests(unittest.TestCase):
    def test_hidden_reference_preserves_loss(self):
        high,c,r,y,d=hidden_data(np.random.default_rng(1234),20000,.02)
        np.testing.assert_array_equal(d,(c!=y).astype(float)-(r!=y).astype(float))
        self.assertGreater(r.mean(),.45)
        self.assertLess(r.mean(),.55)

    def test_collapse_same_moment_and_labels_lower_reference_cost(self):
        rng=np.random.default_rng(9042)
        for _ in range(100):
            q=rng.uniform(.1,1,2);l=rng.uniform(.1,1,2)
            np.testing.assert_allclose(moment(q,l),moment(q*l,np.ones(2)))
            self.assertLessEqual(cost(q*l,np.ones(2),5),cost(q,l,5)+1e-12)
            np.testing.assert_allclose(W*q*G*l,W*(q*l)*G)

    def test_optimized_nested_matches_collapsed(self):
        rng=np.random.default_rng(304)
        for _ in range(20):
            g=rng.uniform(.01,.9,2);cy=float(rng.uniform(.1,10))
            budget=.6*np.sum(W*(1+cy*g))
            q,l=nested_opt(g,cy,budget);qc=collapsed_opt(g,cy,budget)
            self.assertAlmostEqual(moment(q,l,g),moment(qc,np.ones(2),g),places=5)
            self.assertTrue(np.all(l> .999))

    def test_pathwise_coupling(self):
        # Common Bernoulli draws yield identical signed observations, not merely
        # matching estimates on average; collapsed acquisition saves R calls.
        rng=np.random.default_rng(774)
        high,c,r,y,d=generate(rng,10000,.02)
        q=np.where(high,.9,.25);l=np.where(high,1.,.5)
        u=rng.random(len(d));v=rng.random(len(d))
        qr=u<q; purchased=qr & (v<l)
        labels=purchased & (c!=r)
        nested=np.where(labels,d/(q*l),0)
        # purchased ~ Bern(q*l), conditionally independent of R,Y given X.
        collapsed=np.where(purchased & (c!=r),d/(q*l),0)
        np.testing.assert_array_equal(nested,collapsed)
        self.assertLessEqual(int(purchased.sum()),int(qr.sum()))


def simulate_policy(rng,high,c,r,y,q,l,cy,scout_cost):
    prob=q[high.astype(int)];lp=l[high.astype(int)]
    qr=rng.random(len(y))<prob
    observed_dis=np.zeros(len(y),dtype=bool)
    observed_dis[qr]=c[qr]!=r[qr]
    qy=qr & observed_dis & (rng.random(len(y))<lp)
    z=np.zeros(len(y));z[qy]=((c[qy]!=y[qy]).astype(float)-(r[qy]!=y[qy]).astype(float))/(prob[qy]*lp[qy])
    charges=qr.astype(float)+cy*qy
    spent=np.cumsum(charges)+scout_cost
    # Stop before any transaction could exceed the budget.
    eligible=np.flatnonzero(spent<=PROTOCOL['total_cost_cap']-(1+cy))
    horizon=int(eligible[-1]+1) if len(eligible) else 0
    if horizon==0:return {'certified':False,'cost':scout_cost,'ref_calls':0,'labels':0,'arrivals':0}
    bound=float(1/np.min(q*l))
    stakes=np.geomspace(.002,.95/(bound-EPS),24)
    wealth=logsumexp(np.cumsum(np.log1p(stakes[:,None]*(EPS-z[None,:horizon])),axis=1),axis=0)-np.log(24)
    cross=np.flatnonzero(wealth>=np.log(1/DELTA))
    end=int(cross[0]+1) if len(cross) else horizon
    return {'certified':bool(len(cross)),'cost':float(spent[end-1]),
      'ref_calls':int(qr[:end].sum()),'labels':int(qy[:end].sum()),'arrivals':end}


def main():
    out=ROOT/'results'/'staged_audit';out.mkdir(parents=True,exist_ok=True)
    protocol_path=out/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==PROTOCOL
    else:protocol_path.write_text(json.dumps(PROTOCOL,indent=2))
    start=time.perf_counter()
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests))
    if not tests.wasSuccessful():raise SystemExit(1)
    rows=[];optimization=[]
    for cy in PROTOCOL['label_costs']:
        per_budget=.6*cost(np.ones(2),np.ones(2),cy)
        qo,lo=nested_opt(G,cy,per_budget)
        qc=collapsed_opt(G,cy,per_budget)
        optimization.append({'label_cost':cy,'nested_q':qo.tolist(),'nested_l':lo.tolist(),'collapsed_q':qc.tolist(),
         'nested_second_moment':moment(qo,lo),'collapsed_second_moment':moment(qc,np.ones(2))})
        for target in PROTOCOL['target_excess']:
            for seed in PROTOCOL['seeds']:
                for rep in range(PROTOCOL['reps_per_seed']):
                    rng=np.random.default_rng(seed+rep*100000+int(target*10000)+int(cy*10))
                    hs,cs,rs,ys,ds=hidden_data(rng,PROTOCOL['n_scout'],target)
                    # Learn disagreement rates from paid reference predictions;
                    # no scout labels are needed or charged.
                    scout_cost=len(hs)
                    gh=np.array([(1+np.sum((cs!=rs)&(hs==h)))/(2+np.sum(hs==h)) for h in [False,True]])
                    qlearn=collapsed_opt(gh,cy,.6*cost(np.ones(2),np.ones(2),cy,gh))
                    high,c,r,y,d=hidden_data(rng,PROTOCOL['max_stream'],target)
                    policies={'all_reference':(np.ones(2),np.ones(2),0.),
                      'uniform_reference':(np.full(2,.6),np.ones(2),0.),
                      'oracle_joint':(qo,lo,0.),'oracle_collapsed':(qc,np.ones(2),0.),
                      'learned_allocation':(qlearn,np.ones(2),float(scout_cost)),
                      'smoke_joint':(np.array([.25,.9]),np.array([.5,1.]),0.),
                      'smoke_collapsed':(np.array([.125,.9]),np.ones(2),0.)}
                    for mi,(method,(q,l,sc)) in enumerate(policies.items()):
                        # Same random stream for oracle nested/collapsed (l=1);
                        # identical decisions expected up to optimizer tolerance.
                        method_seed=0 if method.startswith('oracle_') else mi
                        draw=np.random.default_rng(seed+rep*100000+method_seed*1000)
                        value=simulate_policy(draw,high,c,r,y,q,l,cy,sc)
                        rows.append({'label_cost':cy,'true_excess':target,'seed':seed,'rep':rep,'method':method,
                          'scout_cost':sc,'q':q.tolist(),'l':l.tolist(),**value})
            print('completed',cy,target,flush=True)
    summary=[]
    methods=sorted(set(r['method'] for r in rows))
    for cy in PROTOCOL['label_costs']:
        for target in PROTOCOL['target_excess']:
            for method in methods:
                group=[r for r in rows if r['label_cost']==cy and r['true_excess']==target and r['method']==method]
                k=sum(r['certified'] for r in group);n=len(group)
                summary.append({'label_cost':cy,'true_excess':target,'method':method,'n':n,'certification_rate':k/n,
                  'rate_ci95':[0 if k==0 else float(beta.ppf(.025,k,n-k+1)),1 if k==n else float(beta.ppf(.975,k+1,n-k))],
                  'mean_total_cost':float(np.mean([r['cost'] for r in group])),
                  'mean_reference_calls':float(np.mean([r['ref_calls']+PROTOCOL['n_scout']*(method=='learned_allocation') for r in group])),
                  'mean_labels_excluding_scout':float(np.mean([r['labels'] for r in group]))})
    result={'seconds_excluding_imports':time.perf_counter()-start,'tests_passed':tests.testsRun,
      'optimization':optimization,'summary':summary,
      'novelty_gate_passed':False,
      'decision':'Do not write a complete research paper. Context-only two-stage acquisition collapses to one-stage; broader joint allocation has close prior work. No new technical result established.'}
    (out/'raw.json').write_text(json.dumps(rows,indent=2));(out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
