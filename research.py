"""CPU-only reproducible pilot, fresh validation and gated final evaluation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR',os.path.join(os.path.dirname(__file__),'.mpl-cache'))
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
from scipy.stats import t
from sklearn.datasets import load_digits, load_breast_cancer, load_wine, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, log_loss
from certificates import METHODS, choose

ROOT = Path(__file__).resolve().parent
EPS, DELTA = .05, .05
THRESHOLDS = np.linspace(.99,.5,12)  # fixed, confidence from cheap model only
PROTOCOL = {'epsilon':EPS,'delta':DELTA,'thresholds':THRESHOLDS.tolist(),
 'primary':'mean pilot routed fraction relative to strongest valid baseline',
 'gate':'candidate paired mean routed-fraction gain >= .03, positive in >= 2/3 seeds, and greater than each component ablation; observed excess error <= .05',
 'pilot_seeds':[11,22,33], 'fresh_seeds':[101,202,303],
 'full_seeds':list(range(1000,1020)), 'datasets':['digits','cancer','wine','synthetic'],
 'split_fractions':[.40,.10,.20,.10,.10,.10],
 'interpretation':'Fixed global pools: train+tune, calibration, pilot, fresh, final. Only train/tune assignment varies by seed. Fresh and final rows never enter pilot fitting/calibration/evaluation. Final 10% never transformed or predicted until gated full run. No tuning on calibration labels.'}


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


def data(name):
    if name == 'synthetic':
        return make_classification(n_samples=12000,n_features=24,n_informative=12,
             n_redundant=6,class_sep=1.2,flip_y=.03,random_state=712)
    return {'digits':load_digits,'cancer':load_breast_cancer,'wine':load_wine}[name](return_X_y=True)


def partitions(n, seed):
    # IID random partition (not stratified, consistent with certificate assumptions).
    ix = np.random.default_rng(904).permutation(n)
    pool,cal,dev,fresh,final=np.split(ix,(np.array([.5,.7,.8,.9])*n).astype(int))
    pool=np.random.default_rng(seed).permutation(pool)
    train,tune=np.split(pool,[int(.4*n)])
    return train,tune,cal,dev,fresh,final


def fit_models(x,y,train,tune,seed):
    models,meta = [],[]
    for kind,grid in [('cheap',[.1,1.,10.]),('reference',[1.,10.,100.])]:
        best = None
        start = time.perf_counter()
        scores=[]
        for c in grid:
            classifier = (LogisticRegression(C=c,max_iter=1500,random_state=seed)
                          if kind=='cheap' else SVC(C=c,random_state=seed))
            m = make_pipeline(StandardScaler(),classifier).fit(x[train],y[train])
            acc = accuracy_score(y[tune],m.predict(x[tune]))
            scores.append(float(acc))
            if best is None or acc > best[0]:
                best = (acc,c,m)
        models.append(best[2])
        meta.append({'kind':kind,'C':best[1],'tune_scores':scores,
                     'training_and_tuning_seconds':time.perf_counter()-start,
                     'grid_size':3})
    return *models,meta


def outputs(cheap,ref,x,y,indices):
    p = cheap.predict_proba(x[indices])
    a = cheap.classes_[p.argmax(1)]
    b = ref.predict(x[indices])
    route = p.max(1)[:,None] >= THRESHOLDS[None,:]
    d = ((a!=y[indices]).astype(int)-(b!=y[indices]).astype(int))[:,None]*route
    return d,route,a,b,p


def latency(m,x,proba=False):
    fn = m.predict_proba if proba else m.predict
    fn(x)
    measurements=[]
    for _ in range(7):
        start=time.perf_counter(); fn(x)
        measurements.append((time.perf_counter()-start)/len(x))
    return float(np.median(measurements))


def run(args):
    out=ROOT/'results'/args.stage
    out.mkdir(parents=True,exist_ok=True)
    if args.stage=='full':
        gate=ROOT/'results'/'pilot'/'gate.json'
        if not gate.exists() or not json.loads(gate.read_text())['pass']:
            raise SystemExit('Full evaluation denied: pilot gate has not passed.')
        validation=ROOT/'results'/'fresh'/'gate.json'
        if not validation.exists() or not json.loads(validation.read_text())['pass']:
            raise SystemExit('Full evaluation denied: fresh confirmation gate has not passed.')
    seeds=PROTOCOL[{'pilot':'pilot_seeds','fresh':'fresh_seeds','full':'full_seeds'}[args.stage]]
    rows=[]; manifests=[]
    for name in PROTOCOL['datasets']:
        x,y=data(name)
        for seed in seeds:
            train,tune,cal,dev,fresh,final=partitions(len(y),seed)
            cheap,ref,training=fit_models(x,y,train,tune,seed)
            # Stage full alone is allowed to select final rows for evaluation.
            evaluate={'full':final,'fresh':fresh,'pilot':dev}[args.stage]
            dcal,_,_,_,_=outputs(cheap,ref,x,y,cal)
            deval,route,a,b,p=outputs(cheap,ref,x,y,evaluate)
            cheap_s=latency(cheap,x[tune],True); ref_s=latency(ref,x[tune])
            manifest={'dataset':name,'seed':seed,'data_sha256':hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest(),
                      'splits':dict(zip(['train','tune','cal','dev','fresh','final'],[z.tolist() for z in [train,tune,cal,dev,fresh,final]])),
                      'training':training,'cheap_batch_seconds_per_row':cheap_s,
                      'ref_batch_seconds_per_row':ref_s,'protocol_hash':digest(PROTOCOL)}
            manifests.append(manifest)
            for method in METHODS:
                for correction in ['fixed','bonferroni']:
                    start=time.perf_counter()
                    j=choose(dcal,EPS,DELTA,method,correction)
                    cert_s=time.perf_counter()-start
                    fraction=float(route[:,j].mean()) if j>=0 else 0.
                    excess=float(deval[:,j].mean()) if j>=0 else 0.
                    pred=np.where(route[:,j],a,b) if j>=0 else b
                    rows.append({'dataset':name,'seed':seed,'method':method,'correction':correction,
                        'selected_index':j,'n_cal':len(cal),'n_eval':len(evaluate),
                        'route_fraction':fraction,'excess_error':excess,
                        'reference_accuracy':float(np.mean(b==y[evaluate])),
                        'accuracy':float(np.mean(pred==y[evaluate])),
                        'cheap_accuracy':float(np.mean(a==y[evaluate])),
                        'cheap_nll':float(log_loss(y[evaluate],p,labels=cheap.classes_)),
                        'estimated_latency_ratio':(cheap_s+(1-fraction)*ref_s)/ref_s if j>=0 else 1.,
                        'certificate_seconds':cert_s})
            # Robustness checks: no distribution-free guarantee under shift.
            rng=np.random.default_rng(seed+50000)
            scale=np.std(x[train],axis=0)
            for severity in [.1,.3]:
                shifted=x[evaluate]+rng.normal(size=x[evaluate].shape)*scale*severity
                ds,rs,_,_,_=outputs(cheap,ref,shifted,y[evaluate],np.arange(len(evaluate)))
                for method in METHODS:
                    j=choose(dcal,EPS,DELTA,method)
                    rows.append({'dataset':name,'seed':seed,'method':method,'correction':f'shift_{severity}',
                      'route_fraction':float(rs[:,j].mean()) if j>=0 else 0.,
                      'excess_error':float(ds[:,j].mean()) if j>=0 else 0.})
            print(args.stage,name,seed,'done',flush=True)
    (out/'raw.json').write_text(json.dumps(rows,indent=2))
    (out/'manifest.json').write_text(json.dumps({'protocol':PROTOCOL,'runs':manifests,
        'python':sys.version,'platform':platform.platform()},indent=2))
    aggregate(out)


def aggregate(out):
    rows=json.loads((out/'raw.json').read_text())
    keys=sorted(set((r['dataset'],r['method'],r['correction']) for r in rows))
    summary=[]
    for dataset,method,correction in keys:
        group=[r for r in rows if (r['dataset'],r['method'],r['correction'])==(dataset,method,correction)]
        v=np.array([r['route_fraction'] for r in group]); n=len(v)
        half=float(t.ppf(.975,n-1)*v.std(ddof=1)/np.sqrt(n)) if n>1 else None
        summary.append({'dataset':dataset,'method':method,'correction':correction,'n':n,
                        'route_mean':float(v.mean()),'route_ci95_halfwidth':half,
                        'excess_error_mean':float(np.mean([r['excess_error'] for r in group])),
                        'empirical_excess_above_epsilon_count':sum(r['excess_error']>EPS for r in group)})
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    with (out/'table.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=summary[0].keys());writer.writeheader();writer.writerows(summary)
    # Each dataset contributes equally; each seed is paired across methods.
    fixed=[r for r in rows if r['correction']=='fixed']
    means={m:np.mean([r['route_fraction'] for r in fixed if r['method']==m]) for m in METHODS}
    baseline=max([m for m in METHODS if m not in ['ternary_universal','naive']],key=lambda m:means[m])
    gains=[]
    for r in fixed:
        if r['method']=='ternary_universal':
            b=next(q for q in fixed if q['dataset']==r['dataset'] and q['seed']==r['seed'] and q['method']==baseline)
            gains.append(r['route_fraction']-b['route_fraction'])
    candidate=[r for r in fixed if r['method']=='ternary_universal']
    passed=bool(np.mean(gains)>=.03 and np.mean(np.array(gains)>0)>=2/3
                and max(r['excess_error'] for r in candidate)<=EPS
                and means['ternary_universal']>means['harm_only']
                and means['ternary_universal']>means['mixture_betting'])
    gate={'pass':passed,'strongest_baseline':baseline,'means':{k:float(v) for k,v in means.items()},
          'paired_primary_gain':float(np.mean(gains)),'positive_fraction':float(np.mean(np.array(gains)>0)),
          'decision':'eligible for next stage' if passed else 'NO full experiment: candidate did not meet predefined gate',
          'warning':'Three seeds give imprecise intervals. Empirical test excess is not population risk; violation counts are descriptive.'}
    (out/'gate.json').write_text(json.dumps(gate,indent=2))
    # Paper-ready source table; include uncertainty without clipping intervals.
    lines=['Dataset & Method & Routed fraction (95\\% t interval) & Excess error \\\\','\\hline']
    for r in summary:
        if r['correction']=='fixed':
            lines.append(f"{r['dataset']} & {r['method'].replace('_', ' ')} & {r['route_mean']:.3f} $\\pm$ {r['route_ci95_halfwidth']:.3f} & {r['excess_error_mean']:.3f} \\\\")
    (out/'table.tex').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(range(len(METHODS)),[means[m] for m in METHODS])
    ax.set_xticks(range(len(METHODS)),METHODS,rotation=30,ha='right')
    ax.set_ylabel('Mean routed fraction (development evaluation)')
    ax.set_ylim(0,1);fig.tight_layout()
    fig.savefig(out/'routing.pdf');fig.savefig(out/'routing.png',dpi=180);plt.close(fig)
    print(json.dumps(gate,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run');r.add_argument('--stage',choices=['pilot','fresh','full'],default='pilot')
    a=sub.add_parser('aggregate');a.add_argument('--stage',default='pilot',choices=['pilot','fresh','full'])
    args=p.parse_args()
    if args.command=='run':run(args)
    else:aggregate(ROOT/'results'/args.stage)
