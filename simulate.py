"""Controlled IID size/power diagnostic, independent of real final test rows."""
import json
import argparse
import numpy as np
from scipy.stats import beta
from certificates import METHODS,certified
from research import ROOT,EPS,DELTA


def main(reps):
    rows=[]
    for n in [250,1000,4000]:
        for q in [.06,.2,.6]:
            for mean in [0.,.04,.05,.06]:
                if q<mean:continue
                # Same simulated outcomes across methods, all seeds retained.
                counts=dict.fromkeys(METHODS,0)
                for seed in [51,52,53]:
                    rng=np.random.default_rng(seed+n+int(q*1000)+int(mean*10000))
                    for _ in range(reps):
                        d=rng.choice([-1.,0.,1.],n,p=[(q-mean)/2,1-q,(q+mean)/2])
                        for m in METHODS:counts[m]+=int(certified(d,EPS,DELTA,m))
                total=3*reps
                for m,k in counts.items():
                    rows.append({'n':n,'discordance':q,'true_excess':mean,'method':m,
                      'rejections':k,'trials':total,'rate':k/total,
                      'ci95_low':0. if k==0 else float(beta.ppf(.025,k,total-k+1)),
                      'ci95_high':1. if k==total else float(beta.ppf(.975,k+1,total-k))})
                print('simulation',n,q,mean,flush=True)
    out=ROOT/'results'/'simulation';out.mkdir(parents=True,exist_ok=True)
    (out/'raw.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reps-per-seed',type=int,default=100)
    main(p.parse_args().reps_per_seed)
