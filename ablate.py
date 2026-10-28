"""Post-confirmation component diagnosis on DEVELOPMENT ONLY; no gate tuning."""
import json
import numpy as np
from research import ROOT,PROTOCOL,EPS,DELTA,data,partitions,fit_models,outputs
from certificates import universal_log_e

rows=[]
for name in PROTOCOL['datasets']:
    x,y=data(name)
    for seed in PROTOCOL['pilot_seeds']:
        train,tune,cal,dev,fresh,final=partitions(len(y),seed)
        cheap,ref,_=fit_models(x,y,train,tune,seed)
        dc,*_=outputs(cheap,ref,x,y,cal)
        de,route,*_=outputs(cheap,ref,x,y,dev)
        for method in ['full','no_benefit_credit','flat_prior_only']:
            selected=-1
            for j in range(dc.shape[1]):
                d=np.maximum(dc[:,j],0) if method=='no_benefit_credit' else dc[:,j]
                priors=[np.array([.5,.5,.5])] if method=='flat_prior_only' else None
                if universal_log_e(d,EPS,priors=priors)>=np.log(1/DELTA):selected=j
                else:break
            rows.append({'dataset':name,'seed':seed,'ablation':method,
                         'route_fraction':float(route[:,selected].mean()) if selected>=0 else 0.,
                         'excess_error':float(de[:,selected].mean()) if selected>=0 else 0.})
        print('ablation',name,seed,flush=True)
out=ROOT/'results'/'ablations';out.mkdir(parents=True,exist_ok=True)
(out/'raw.json').write_text(json.dumps(rows,indent=2))
means={m:float(np.mean([r['route_fraction'] for r in rows if r['ablation']==m])) for m in ['full','no_benefit_credit','flat_prior_only']}
(out/'summary.json').write_text(json.dumps({'means':means,'status':'Post-confirmation diagnostic on original development rows only; not confirmation of a selected improvement.'},indent=2))
print(means)
