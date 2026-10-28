"""Post-pilot diagnostic alternatives. All use development/tuning labels only.

These are known universal-inference/betting specializations, not new methods.
Run only after reading the original pilot. No threshold/risk changes allowed.
"""
import argparse
import json
import time
import numpy as np
from certificates import universal_log_e, choose, METHODS
from research import ROOT, PROTOCOL, EPS, DELTA, data, partitions, fit_models, outputs


def select(dc, dt, method):
    selected=-1
    for j in range(dc.shape[1]):
        if method=='tune_likelihood':
            counts=np.array([(dt[:,j]==v).sum() for v in [-1,0,1]])+.5
            log_e=universal_log_e(dc[:,j],EPS,counts/counts.sum())
        else:
            stakes=np.r_[0.,np.geomspace(.002,.95/(1-EPS),40)]
            s=np.argmax(np.log1p(stakes[:,None]*(EPS-dt[:,j])).sum(axis=1))
            log_e=float(np.log1p(stakes[s]*(EPS-dc[:,j])).sum())
        if log_e >= np.log(1/DELTA):selected=j
        else:break
    return selected


def run(stage):
    out=ROOT/'results'/f'alternative_{stage}';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    seeds=PROTOCOL['pilot_seeds' if stage=='pilot' else 'fresh_seeds']
    for name in PROTOCOL['datasets']:
        x,y=data(name)
        for seed in seeds:
            train,tune,cal,dev,fresh,final=partitions(len(y),seed)
            cheap,ref,meta=fit_models(x,y,train,tune,seed)
            dc,*_=outputs(cheap,ref,x,y,cal)
            dt,*_=outputs(cheap,ref,x,y,tune)
            de,route,*_=outputs(cheap,ref,x,y,dev if stage=='pilot' else fresh)
            for method in METHODS+['tune_likelihood','tune_betting']:
                start=time.perf_counter()
                j=select(dc,dt,method) if method.startswith('tune_') else choose(dc,EPS,DELTA,method)
                rows.append({'dataset':name,'seed':seed,'method':method,'selected_index':j,
                  'route_fraction':float(route[:,j].mean()) if j>=0 else 0.,
                  'excess_error':float(de[:,j].mean()) if j>=0 else 0.,
                  'certificate_seconds':time.perf_counter()-start})
            print('alternative',stage,name,seed,flush=True)
    (out/'raw.json').write_text(json.dumps(rows,indent=2))
    means={m:float(np.mean([r['route_fraction'] for r in rows if r['method']==m])) for m in METHODS+['tune_likelihood','tune_betting']}
    summary={'means':means,'interpretation':'Exploratory alternatives. No novel method or final-test claim. Same epsilon, delta, thresholds and pools.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','fresh'],default='pilot')
    run(p.parse_args().stage)
