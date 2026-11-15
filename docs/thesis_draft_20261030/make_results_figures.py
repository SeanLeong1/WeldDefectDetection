"""Read-only verification and report figures from frozen confirmation evidence."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'DataSimulation'),str(ROOT/'WeldDefectDetection')]
from count_study import campaign as c,learning as learn,reporting
from circular_study import metrics
from raised_crown import learning as base

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def aggregate(rows):
    tp,fp,fn=[sum(r[k] for r in rows) for k in ('tp','fp','fn')]
    clean=[r for r in rows if r['source_class']=='clean']
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1),
        pixel_iou=sum(r['intersection'] for r in rows)/max(sum(r['union'] for r in rows),1),
        clean_rings=len(clean),clean_rings_flagged=sum(r['predicted_regions']>0 for r in clean),
        clean_ring_fa=sum(r['predicted_regions']>0 for r in clean)/max(len(clean),1),
        clean_false_regions=sum(r['fp'] for r in clean),scenes=len(rows))

def main():
    c.verify();learn.verify_development()
    figdir=HERE/'figures';ev=HERE/'evidence';figdir.mkdir(exist_ok=True);ev.mkdir(exist_ok=True)
    files=['results.json','frozen_models.json','paired_comparisons.json','measured_noise.json','diagnostic_results.json',
           'normalization.json','independent_final_audit.json','final_delivery_verification.json','noise_checks.json','method_checks.json','datasets/test/index.json']
    source={n:c.read(c.OUT/n) for n in files};hashes={str((c.OUT/n).relative_to(ROOT)):sha(c.OUT/n) for n in files}
    results=source['results.json'];frozen=source['frozen_models.json'];paired=source['paired_comparisons.json'];noise=source['measured_noise.json']
    index=source['datasets/test/index.json'];lookup={r['scene_id']:i for i,r in enumerate(index)}
    chosen=reporting.selected_results(results);checks=0
    for name,item in results.items():
        assert len(item['rows'])==864 and {r['scene_id'] for r in item['rows']}==set(lookup)
        groups=[(item['rows'],item['metrics'])]
        groups += [([r for r in item['rows'] if r['count_multiplier']==m],item['by_count'][str(m)]) for m in c.MULTIPLIERS]
        groups += [([r for r in item['rows'] if r['source_class']=='clean' or lo<=r['diameter_mm']<hi],item['by_size'][label]) for label,lo,hi in [('small',.15,.30),('medium',.30,.60),('large',.60,1.01)]]
        for rows,saved in groups:
            for key,value in aggregate(rows).items():assert np.isclose(saved[key],value,rtol=0,atol=1e-12),(name,key)
            checks+=1
        if item['kind']=='cnn' and item['arm']!='retained':assert item['selected']==(item['seed']==frozen['arms'][item['arm']]['cnn']['seed'])
    for m,comparisons in paired.items():
        for label,value in comparisons.items():
            a,b=label.split(' minus ')
            rec=reporting.paired([r for r in chosen[a]['rows'] if r['count_multiplier']==int(m)],
                                 [r for r in chosen[b]['rows'] if r['count_multiplier']==int(m)])
            assert rec==value,(m,label)
    # Recheck clean repeat variability against raw acquisitions, without touching study files.
    clean=[r for r in noise if r['source_class']=='clean']
    for r in clean:
        scene=f"{r['background_id']}-clean-c{r['count_multiplier']}-r"
        a,b=[index[lookup[scene+str(rep)]] for rep in (0,1)]
        arrays=[]
        for row in (a,b):
            p=Path(row['raw_path']);assert sha(p)==row['raw_sha256']
            hashes[str(p.relative_to(ROOT))]=row['raw_sha256'];arrays.append(np.load(p,mmap_mode='r')[16:144])
        observed=float(np.sqrt(np.mean((arrays[0]-arrays[1])**2)/2))
        assert observed==r['repeat_difference_rms_over_sqrt2']
    by={(x['background_id'],x['count_multiplier']):x['repeat_difference_rms_over_sqrt2'] for x in clean}
    keys=sorted({x['background_id'] for x in clean})
    ratios={str(m):float(np.median([by[k,m]/by[k,1] for k in keys])) for m in (4,64)}
    medians={str(m):float(np.median([by[k,m] for k in keys])) for m in c.MULTIPLIERS}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    def save(fig,name):
        for ext in ['png','pdf','svg']:fig.savefig(figdir/f'{name}.{ext}',dpi=210,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for b,label in enumerate(['0.1–0.5 µs','0.5–2 µs','2–8 µs','8–20 µs']):
        vals=[np.median([x['repeat_difference_rms_over_sqrt2'] for x in clean if x['count_multiplier']==m and x['dwell_bin']==b]) for m in c.MULTIPLIERS]
        axes[0].plot(c.MULTIPLIERS,vals,'o-',label=label)
    axes[0].set(xscale='log',yscale='log',xticks=[1,4,64],xticklabels=['1','4','64'],xlabel='Effective-count multiplier',ylabel='Repeat-difference RMS / √2 (raw TOPO units)',title='(a) Median across 12 clean backgrounds per dwell bin')
    axes[0].legend(title='Dwell interval',fontsize=9);axes[0].grid(alpha=.2)
    for k in keys:axes[1].plot([1,4,64],[1,by[k,4]/by[k,1],by[k,64]/by[k,1]],color='#7395ac',alpha=.16,lw=.8)
    axes[1].plot([1,4,64],[1,ratios['4'],ratios['64']],'o-',color='#153f65',lw=2.5,label='Median paired ratio')
    axes[1].plot([1,4,64],[1,.5,.125],'s--',color='#b95130',label='Shot-only 1/√m reference')
    axes[1].set(xscale='log',yscale='log',xticks=[1,4,64],xticklabels=['1','4','64'],xlabel='Effective-count multiplier',ylabel='Variability / same background at 1×',title='(b) Paired ratios across 48 clean backgrounds')
    axes[1].legend(fontsize=9);axes[1].grid(alpha=.2)
    save(fig,'figure_5_1_noise')
    arms=list(c.ARMS);labels=['1× specialist, 96','4× specialist, 96','64× specialist, 96','Mixed, 96','1× specialist, 24','Mixed, 24']
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,method in zip(axes,['median','cnn']):
        values=np.array([[chosen[f'{a}_{method}']['by_count'][str(m)]['f1'] for m in c.MULTIPLIERS] for a in arms])
        im=ax.imshow(values,vmin=0,vmax=1,cmap='viridis',aspect='auto')
        ax.set(yticks=range(6),yticklabels=labels,xticks=range(3),xticklabels=['1×','4×','64×'],xlabel='Confirmation counts',title='Neighbouring median' if method=='median' else 'TinyDefectSegNet')
        for (y,x),v in np.ndenumerate(values):ax.text(x,y,f'{v:.3f}',ha='center',va='center',color='white' if v<.55 else 'black')
    fig.colorbar(im,ax=axes,label='Detection F1',shrink=.8);save(fig,'figure_5_2_transfer')
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),layout='constrained',sharey=True)
    for ax,m in zip(axes,c.MULTIPLIERS):
        for j,arm in enumerate(arms):
            candidates=[v for v in results.values() if v['arm']==arm and v['kind']=='cnn']
            vals=[v['by_count'][str(m)]['f1'] for v in candidates]
            ax.plot([min(vals),max(vals)],[j,j],color='#93a4b5',lw=2)
            ax.scatter(vals,[j]*3,color='#859db4',s=26)
            ax.scatter(chosen[arm+'_cnn']['by_count'][str(m)]['f1'],j,marker='D',color='#bc512f',s=45,zorder=3)
        ax.set(xlim=(0,1),xticks=[0,.25,.5,.75,1],yticks=range(6),yticklabels=labels,xlabel='Detection F1',title=f'{m}× confirmation');ax.grid(axis='x',alpha=.2)
    axes[0].invert_yaxis();fig.legend(handles=[Line2D([],[],marker='o',color='#859db4',ls='',label='Three training seeds'),Line2D([],[],marker='D',color='#bc512f',ls='',label='Validation-selected seed')],loc='outside lower center',ncol=2)
    save(fig,'figure_5_3_seeds')

    scorefiles={'cnn':c.OUT/'confirmation'/f"mixed_96_cnn_{frozen['arms']['mixed_96']['cnn']['seed']}.npy",'median':c.OUT/'confirmation/mixed_96_median.npy'}
    scores={k:np.load(p,mmap_mode='r') for k,p in scorefiles.items()}
    savedrows={k:{r['scene_id']:r for r in chosen['mixed_96_'+k]['rows']} for k in scores}
    norm=source['normalization.json'];rendered={}
    def scene(sid):
        if sid in rendered:return rendered[sid]
        row=index[lookup[sid]]
        for pathkey,hashkey in [('raw_path','raw_sha256'),('geometry_path','geometry_sha256')]:
            p=Path(row[pathkey]);assert sha(p)==row[hashkey];hashes[str(p.relative_to(ROOT))]=sha(p)
        with np.load(row['geometry_path']) as z:truth=z['mask'][16:144]
        x=base.normalize(np.load(row['raw_path']),norm)[16:144];pred={};summary={}
        for kind in scores:
            op=frozen['arms']['mixed_96'][kind]['selection']['best']
            mask,boxes=metrics.binary(scores[kind][lookup[sid]],op['threshold'],op['min_area'],op['closing']);truthboxes=metrics.regions(truth)[1]
            calc=metrics.match(boxes,truthboxes,5024);record=savedrows[kind][sid]
            for key,value in calc.items():assert np.isclose(record[key],value)
            assert int(mask.sum())==record['predicted_pixels'] and int(((mask>0)&(truth>0)).sum())==record['intersection']
            summary[kind]={k:record[k] for k in ['tp','fp','fn','predicted_regions','truth_pixels','intersection','union']}
            summary[kind]['best_box_iou']=max([metrics.circular_iou(a,b,5024) for a in boxes for b in truthboxes] or [0.])
            pred[kind]=mask
        rendered[sid]=(x,truth,pred,summary);return rendered[sid]
    def window(truth,width=64):
        box=metrics.regions(truth)[1][0];cx=int(round((box[0]+box[2])/2))%5024;cy=int(round((box[1]+box[3])/2))
        y0=min(max(0,cy-width//2),128-width);return slice(y0,y0+width),(np.arange(width)+cx-width//2)%5024
    fixed='test-000-open_intrinsic-c1-r0';_,truth,_,_=scene(fixed);rs,cs=window(truth)
    fig,axes=plt.subplots(2,3,figsize=(11.5,8),layout='constrained')
    for y,kind in enumerate(['clean','open_intrinsic']):
        for col,m in enumerate(c.MULTIPLIERS):
            sid=f'test-000-{kind}-c{m}-r0';x,t,p,s=scene(sid);ax=axes[y,col]
            ax.imshow(x[rs][:,cs],origin='lower',cmap='gray',vmin=0,vmax=1)
            for mask,color,style in [(t,'#66de3d','--'),(p['median'],'#ee973b','-'),(p['cnn'],'#31cfe5','-')]:
                a=mask[rs][:,cs]
                if a.any():ax.contour(a,levels=[.5],colors=[color],linestyles=style,linewidths=1.25)
            counts=lambda k:'/'.join(str(s[k][z]) for z in ['tp','fp','fn'])
            ax.set(title=f'{"Clean" if y==0 else "Intrinsic pore"}, {m}×\nCNN {counts("cnn")}; median {counts("median")}',xlabel='Local angular pixels',ylabel='Local radial pixels')
    fig.suptitle('Fixed confirmation background test-000, repeat 0\nTitles give full-ring TP/FP/FN; panels show the same local crop',fontsize=12)
    fig.legend(handles=[Line2D([],[],color='#66de3d',ls='--',label='Geometry truth'),Line2D([],[],color='#ee973b',label='Median'),Line2D([],[],color='#31cfe5',label='CNN')],loc='outside lower center',ncol=3)
    save(fig,'figure_5_4_matched_examples')
    large=next(r['scene_id'] for r in chosen['mixed_96_median']['rows'] if r['source_class']!='clean' and r['diameter_mm']>=.6 and r['count_multiplier']==64 and r['repeat']==0)
    small=next(r['scene_id'] for r in chosen['mixed_96_cnn']['rows'] if r['source_class']!='clean' and r['diameter_mm']<.3 and r['count_multiplier']==64 and r['repeat']==0 and r['fn']>0)
    fig,axes=plt.subplots(2,3,figsize=(11,7.5),layout='constrained')
    for y,sid in enumerate([large,small]):
        x,t,p,s=scene(sid);rs,cs=window(t,32);row=index[lookup[sid]]
        for col,kind in enumerate(['input','median','cnn']):
            ax=axes[y,col];a=(x if kind=='input' else p[kind])[rs][:,cs]
            ax.imshow(a,origin='lower',cmap='gray',vmin=0,vmax=1)
            ax.contour(t[rs][:,cs],levels=[.5],colors=['#45c179'],linestyles='--',linewidths=1.2)
            title='Normalised TOPO' if kind=='input' else f'{kind.upper()} mask; best box IoU {s[kind]["best_box_iou"]:.3f}'
            ax.set(title=title,xlabel='Local angular pixels',ylabel='Local radial pixels')
        axes[y,0].set_ylabel(f'{row["background_id"]}, D={row["diameter_mm"]:.3f} mm\n{row["area_px"]} truth pixels\nLocal radial pixels')
    fig.suptitle('Confirmation error examples at 64× counts, repeat 0; green dashed contour marks truth\nFirst large intrinsic case and first small CNN miss in saved scene order',fontsize=12)
    save(fig,'figure_5_5_failure_examples')
    selected={k:{q:v[q] for q in ['by_count','by_size','prediction_seconds']} for k,v in chosen.items()}
    seed_ranges={a:{str(m):[min(x['by_count'][str(m)]['f1'] for x in results.values() if x['arm']==a and x['kind']=='cnn'),max(x['by_count'][str(m)]['f1'] for x in results.values() if x['arm']==a and x['kind']=='cnn')] for m in c.MULTIPLIERS} for a in c.ARMS}
    receipt=dict(campaign=c.OUT.name,source_hashes=hashes,aggregate_groups_verified=checks,paired_comparisons_verified=sum(map(len,paired.values())),clean_repeat_pairs_verified=len(clean),
                 clean_noise_medians=medians,median_paired_noise_ratios=ratios,selected=selected,seed_ranges=seed_ranges,
                 diagnostic={name:{mode+f'_c{m}':aggregate([r for r in rows if r['mode']==mode and r['count_multiplier']==m]) for mode in ['noiseless','shot_only','full'] for m in ([1] if mode=='noiseless' else c.MULTIPLIERS)} for name,rows in source['diagnostic_results.json'].items()},
                 examples={sid:v[-1] for sid,v in rendered.items()},failure_selection={'first_large':large,'first_small_cnn_miss':small},
                 selected_train_seconds={a:v['cnn']['train_seconds'] for a,v in frozen['arms'].items()},
                 figure_files={p.name:sha(p) for p in sorted(figdir.glob('figure_5_*'))},generator_sha256=sha(__file__),frozen_artifacts_changed=False)
    (ev/'chapter5_results_provenance.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'aggregate_groups_verified':checks,'clean_noise_ratios':ratios,'failure_selection':receipt['failure_selection'],'examples':receipt['examples']},indent=2))

if __name__=='__main__':main()
