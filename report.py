"""Build reports only from saved observations. Never trains or opens test data."""
import json
import csv
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.mpl-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research import ROOT,EPS


def read(path):return json.loads((ROOT/path).read_text())


def main():
    out=ROOT/'results'/'report';out.mkdir(parents=True,exist_ok=True)
    tables=[]
    for stage in ['pilot','fresh']:
        raw=read(f'results/{stage}/raw.json')
        meta=read(f'results/{stage}/manifest.json')
        final_sets={}
        for run in meta['runs']:
            pools=run['splits'];final=set(pools['final'])
            assert all(not final.intersection(pools[k]) for k in ['train','tune','cal','dev','fresh'])
            key=run['dataset']
            if key in final_sets:assert final==final_sets[key]
            final_sets[key]=final
        fixed=[r for r in raw if r['correction']=='fixed']
        for r in fixed:
            assert abs((r['reference_accuracy']-r['accuracy'])-r['excess_error'])<1e-10
        for name in ['digits','cancer','wine','synthetic']:
            for method in ['ternary_universal','harm_only','predictable_betting']:
                group=[r for r in fixed if r['dataset']==name and r['method']==method]
                tables.append({'stage':stage,'dataset':name,'method':method,
                 **{metric:float(np.mean([r[metric] for r in group])) for metric in ['route_fraction','excess_error','accuracy','reference_accuracy','cheap_accuracy','estimated_latency_ratio','certificate_seconds']}})
        shifts={c:{'route':float(np.mean([r['route_fraction'] for r in raw if r['correction']==c and r['method']=='ternary_universal'])),
                   'excess':float(np.mean([r['excess_error'] for r in raw if r['correction']==c and r['method']=='ternary_universal']))}
                for c in ['shift_0.1','shift_0.3']}
        timing=sum(m['training_and_tuning_seconds'] for run in meta['runs'] for m in run['training'])
        audit={'checks':'fixed global final isolation; accuracy/excess identity; all 3 seeds for each dataset',
               'training_tuning_seconds_total':timing,'candidate_shift_macro_metrics':shifts,
               'candidate_empirical_exceedances':sum(r['excess_error']>EPS for r in fixed if r['method']=='ternary_universal'),
               'naive_empirical_exceedances':sum(r['excess_error']>EPS for r in fixed if r['method']=='naive')}
        (out/f'{stage}_audit.json').write_text(json.dumps(audit,indent=2))
    with (out/'verified_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=tables[0]);w.writeheader();w.writerows(tables)
    (out/'verified_comparison.json').write_text(json.dumps(tables,indent=2))
    sim=read('results/simulation/raw.json')
    fig,axes=plt.subplots(1,3,figsize=(11,3.4),sharey=True)
    methods=['ternary_universal','mixture_betting','predictable_betting','harm_only']
    for ax,q in zip(axes,[.06,.2,.6]):
        for method in methods:
            group=sorted([r for r in sim if r['discordance']==q and r['true_excess']==0. and r['method']==method],key=lambda r:r['n'])
            ax.errorbar([r['n'] for r in group],[r['rate'] for r in group],
              yerr=[[r['rate']-r['ci95_low'] for r in group],[r['ci95_high']-r['rate'] for r in group]],
              marker='o',label=method.replace('_',' '),capsize=2)
        ax.set_xscale('log');ax.set_title(f'Discordance probability {q}')
        ax.set_xticks([250,1000,4000],['250','1000','4000'])
        ax.minorticks_off()
        ax.set_xlabel('Calibration samples');ax.set_ylim(-.03,1.03)
    axes[0].set_ylabel('Certification probability')
    fig.suptitle('True excess error = 0; marginal 95% intervals',fontsize=10)
    handles,labels=axes[-1].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=8,loc='lower center',ncol=4)
    fig.tight_layout(rect=(0,.12,1,.94))
    fig.savefig(out/'power.pdf');fig.savefig(out/'power.png',dpi=180);plt.close(fig)
    null=[r for r in sim if r['true_excess']>=EPS and r['method']!='naive']
    (out/'simulation_audit.json').write_text(json.dumps({'max_null_rejection_rate':max(r['rate'] for r in null),
        'max_null_upper_ci':max(r['ci95_high'] for r in null),
        'warning':'Marginal Monte Carlo intervals, not simultaneous bounds or proof; diagnostic only.',
        'number_scenarios':len(sim)//8,'draws_per_scenario':sim[0]['trials']},indent=2))
    print('Verified reports written to',out)


if __name__=='__main__':main()
