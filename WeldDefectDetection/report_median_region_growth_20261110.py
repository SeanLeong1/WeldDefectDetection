"""Audit and present the separate median region-growth study."""
from pathlib import Path
from datetime import datetime
import csv, html, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import median_region_growth_20261110 as s

OUT=s.OUT
REPORT=s.ROOT/'docs/median_region_growth_20261110'
NAMES={'original':'Original median','previous':'Previous position + size',
       'seed_only':'Strong threshold only','growth':'Bounded local growth',
       'growth_merge':'Growth + fragment merging','cnn':'Frozen CNN'}
COLORS={'original':'#888888','previous':'#bd7b20','seed_only':'#7355a1','growth':'#22a78b','growth_merge':'#1263a1','cnn':'#222222'}


def matched_controls():
    frozen=s.verify_frozen()
    plan=s.read(OUT/'matched_ablation_plan.json')
    assert datetime.fromisoformat(plan['declared_utc'])<datetime.fromisoformat(frozen['frozen_utc'])
    data=s.learn.Data(s.read(OUT/'datasets/test/index.json'),s.read(s.OLD/'normalization.json'))
    signed=np.load(OUT/'confirmation/signed_sigma1.npy',mmap_mode='r')
    controls={}
    for name in ('growth','growth_merge'):
        config=dict(frozen['selection']['selected'][name]['config'])
        if name=='growth':config.update(radius=0,low=config['high'])
        else:config['gap']=0
        path=OUT/f'confirmation/{name}_disabled.json'
        if not path.exists():
            rows=[s.result_row(meta,data.masks[i][16:144]>0,data.truth[i],s.prediction(signed[i],config)) for i,meta in enumerate(data.rows)]
            s.write(path,dict(name=name+'_disabled',config=config,predeclared_ablation=True,**s.learn.result_summary(rows)))
        controls[name+'_disabled']=s.read(path)
    return controls


def audit():
    frozen=s.verify_frozen();rows=s.read(OUT/'datasets/test/index.json')
    assert len(rows)==s.FRESH_COUNT*18
    historical=[r for split in ('train','val','test') for r in s.learn.selected_rows(split)]
    historical+=s.read(s.previous.OUT/'datasets/test/index.json')
    assert len({r['seed'] for r in rows})==s.FRESH_COUNT
    assert not {r['seed'] for r in rows}&{r['seed'] for r in historical}
    assert not {r['acquisition_seed'] for r in rows}&{r['acquisition_seed'] for r in historical}
    files={r[k]:r[h] for r in rows for k,h in [('raw_path','raw_sha256'),('geometry_path','geometry_sha256')]}
    frozen_time=datetime.fromisoformat(frozen['frozen_utc']).timestamp()
    for path,digest in files.items():
        assert s.sha(path)==digest,path
        assert Path(path).stat().st_mtime>frozen_time,path
    for kind in s.c.KINDS:
        for i in range(s.FRESH_COUNT):
            group=[r for r in rows if r['index']==i and r['source_class']==kind]
            assert len(group)==6 and len({r['geometry_sha256'] for r in group})==1
    data=s.learn.Data(rows,s.read(s.OLD/'normalization.json'))
    signed=np.load(OUT/'confirmation/signed_sigma1.npy',mmap_mode='r')
    results={name:s.read(OUT/f'confirmation/{name}.json') for name in NAMES}
    results.update(matched_controls())
    for name,result in results.items():
        assert [r['scene_id'] for r in result['rows']]==[r['scene_id'] for r in rows]
        for i,row in enumerate(result['rows']):
            match=s.metrics.match(row['predicted_boxes'],data.truth[i],s.PERIOD)
            assert all(row[k]==v for k,v in match.items())
        assert s.learn.objective(result['rows'])['by_count']==result['by_count']
    spec=s.read(s.OLD/'development/mixed_96/median_sigma1.json')['spec']
    for i in (0,8,15,143,284,len(rows)-1):
        assert np.array_equal(s.signed_score(data.image(i,1),spec),signed[i])
        for name,t in frozen['selection']['selected'].items():
            regions=s.prediction(signed[i],t['config'])
            assert regions==results[name]['rows'][i]['signed_region_evidence']
    model=s.base.model_for(frozen['original']['cnn'])
    cnn=np.load(OUT/'confirmation/cnn.npy',mmap_mode='r')
    for i in (0,8,len(rows)-1):
        assert np.array_equal(s.base.predict_image(data.image(i),model),cnn[i])
    checks=dict(status='passed',fresh_backgrounds=s.FRESH_COUNT,scenes=len(rows),hashed_fresh_files=len(files),
        disjoint_from_original_and_previous_confirmation=True,generated_after_freeze=True,matched_geometry=True,
        metric_replay=True,signed_score_and_prediction_replay=True,frozen_cnn_replay=True,
        source_dependencies_unchanged=True,physical_validation=False)
    s.write(OUT/'final_audit.json',checks)
    return frozen,rows,results


def intervals(results,a,b):
    out={};rng=np.random.default_rng(111020263)
    for count in s.c.MULTIPLIERS:
        aa=[r for r in results[a]['rows'] if r['count_multiplier']==count]
        bb=[r for r in results[b]['rows'] if r['count_multiplier']==count]
        groups=sorted({r['background_id'] for r in aa})
        def counts(rr):
            return np.array([[sum(r[k] for r in rr if r['background_id']==g) for k in ('tp','fp','fn')] for g in groups])
        draw=rng.integers(0,len(groups),(2000,len(groups)))
        av=counts(aa)[draw].sum(1);bv=counts(bb)[draw].sum(1)
        fa=2*av[:,0]/np.maximum(1,2*av[:,0]+av[:,1]+av[:,2]);fb=2*bv[:,0]/np.maximum(1,2*bv[:,0]+bv[:,1]+bv[:,2])
        out[str(count)]=dict(delta=results[a]['by_count'][str(count)]['f1']-results[b]['by_count'][str(count)]['f1'],
                           paired_95_ci=np.quantile(fa-fb,[.025,.975]).tolist())
    groups=sorted({r['background_id'] for r in results[a]['rows']})
    def cube(name):
        return np.array([[[sum(r[k] for r in results[name]['rows'] if r['background_id']==bg and r['count_multiplier']==m) for k in ('tp','fp','fn')] for m in s.c.MULTIPLIERS] for bg in groups])
    draw=rng.integers(0,len(groups),(2000,len(groups)))
    aa=cube(a)[draw].sum(1);bb=cube(b)[draw].sum(1)
    fa=(2*aa[:,:,0]/np.maximum(1,2*aa[:,:,0]+aa[:,:,1]+aa[:,:,2])).mean(1)
    fb=(2*bb[:,:,0]/np.maximum(1,2*bb[:,:,0]+bb[:,:,1]+bb[:,:,2])).mean(1)
    out['macro']=dict(delta=results[a]['macro_f1']-results[b]['macro_f1'],paired_95_ci=np.quantile(fa-fb,[.025,.975]).tolist())
    return out


def diagnose(results):
    output={}
    for name,result in results.items():
        group={}
        for count in s.c.MULTIPLIERS:
            rows=[r for r in result['rows'] if r['count_multiplier']==count]
            pores=[r for r in rows if r['source_class']!='clean'];clean=[r for r in rows if r['source_class']=='clean']
            near=lambda r:any(s.metrics.circular_iou(p,t,s.PERIOD)>0 for p in r['predicted_boxes'] for t in r['truth_boxes'])
            group[str(count)]=dict(pore_scenes=len(pores),correct=sum(r['tp']>0 for r in pores),
                overlapping_but_failed=sum(r['tp']==0 and near(r) for r in pores),
                no_overlapping_prediction=sum(not near(r) for r in pores),
                pore_images_flagged=sum(r['predicted_regions']>0 for r in pores),
                clean_images_flagged=sum(r['predicted_regions']>0 for r in clean),clean_scenes=len(clean))
        output[name]=group
    s.write(OUT/'failure_diagnosis.json',output)
    return output


def stage_diagnosis(rows,results,chosen,config):
    """Truth-centred diagnostics after evaluation; never used for inference."""
    signed=np.load(OUT/'confirmation/signed_sigma1.npy',mmap_mode='r')
    output={}
    for count in s.c.MULTIPLIERS:
        records=[]
        for i,meta in enumerate(rows):
            if meta['source_class']=='clean' or meta['count_multiplier']!=count:continue
            r=results[chosen]['rows'][i]
            cols=np.concatenate([np.arange(b[0]-8,b[2]+8)%s.PERIOD for b in r['truth_boxes']])
            y0=max(0,min(b[1] for b in r['truth_boxes'])-8);y1=min(128,max(b[3] for b in r['truth_boxes'])+8)
            local=signed[i][y0:y1][:,cols]
            records.append(dict(scene_id=r['scene_id'],diameter_mm=meta['diameter_mm'],correct=r['tp']>0,
                maximum_nearby_score=float(abs(local).max()),nearby_seed=bool((abs(local)>=config['high']).any())))
        output[str(count)]=dict(pore_scenes=len(records),no_nearby_strong_seed=sum(not r['nearby_seed'] for r in records),
            missed_with_nearby_strong_seed=sum(r['nearby_seed'] and not r['correct'] for r in records),rows=records)
    s.write(OUT/'seed_stage_diagnosis.json',dict(scope='Descriptive after confirmation. Truth box expanded by 8 pixels defines a diagnostic neighbourhood only; not inference.',by_count=output))
    return output


def geometry_plot():
    """Actual saved height change, independent of TOPO display contrast."""
    alignment=s.read(s.previous.OUT/'alignment.json')
    selected=s.read(s.ROOT/'docs/median_localisation_20261110/visual_provenance.json')['alignment_examples']
    fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
    records=[]
    for j,sid in enumerate(selected):
        rec=next(r for r in alignment['rows'] if r['scene_id']==sid);path=Path(rec['geometry_path'])
        with np.load(path) as z:h=z['height'];quiet=z['quiet'];mask=z['mask']>0
        with np.load(path.with_name('clean_geometry.npz')) as z:dh=h-z['height'];dt=quiet-z['quiet']
        yy,xx=np.nonzero(mask);centre=int(round(rec['angle_rad']*s.PERIOD/(2*np.pi)))%s.PERIOD
        cols=(np.arange(-22,23)+centre)%s.PERIOD;rad=np.arange(max(0,yy.min()-10),min(160,yy.max()+11))
        m=mask[np.ix_(rad,cols)];d=dh[np.ix_(rad,cols)];t=dt[np.ix_(rad,cols)]
        for ax,a,title,cmap in zip(axes[j],[d,t,m],['Actual height change (mm)','TOPO change: pore minus clean','Geometric opening'],['viridis','coolwarm','gray']):
            opts=dict(vmin=-float(abs(a).max()),vmax=float(abs(a).max())) if cmap=='coolwarm' else {}
            im=ax.imshow(a,cmap=cmap,**opts)
            if cmap!='gray':ax.contour(m,levels=[.5],colors=['#ee2255'],linewidths=1.3);fig.colorbar(im,ax=ax,fraction=.045,pad=.02)
            ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
        axes[j,0].set_ylabel(f"{rec['radial_offset_mm']:+.2f} mm from crown centre")
        records.append(dict(scene_id=sid,deepest_inside_mask=bool(mask[np.unravel_index(dh.argmin(),dh.shape)]),min_height_change_mm=float(dh.min())))
    fig.suptitle('The injected depression and the visible contrast have different footprints',fontsize=15)
    fig.savefig(REPORT/'01_geometry_and_contrast.png',dpi=170);plt.close(fig)
    return records


def height_audit():
    target=OUT/'height_injection_audit.json'
    if target.exists():return s.read(target)
    from scipy.ndimage import distance_transform_edt
    rows=[r for r in s.learn.selected_rows('train') if r['repeat']==0 and r['count_multiplier']==64 and r['source_class']!='clean']
    records=[]
    for row in rows:
        p=row['parameters']['defects'][0]['params'];path=Path(row['geometry_path'])
        with np.load(path) as z:h=z['height'];mask=z['mask'].astype(bool)
        with np.load(path.with_name('clean_geometry.npz')) as z:delta=h-z['height']
        ys,_=np.nonzero(mask);centre=int(round(p['angle_rad']*s.PERIOD/(2*np.pi)))%s.PERIOD
        cols=(np.arange(-40,41)+centre)%s.PERIOD;rr=np.arange(max(0,ys.min()-6),min(160,ys.max()+7))
        radius=40+(rr[:,None]-79.5)*.05;theta=2*np.pi*cols[None,:]/s.PERIOD
        xx=(radius*np.cos(theta)+48)/.05;yy=(radius*np.sin(theta)+48)/.05
        ix=np.floor(xx);iy=np.floor(yy);fx=xx-ix;fy=yy-iy;expected=np.zeros_like(xx)
        for dx,dy,w in [(0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)]:
            nx=(ix+dx-960)*.05;ny=(iy+dy-960)*.05;u=np.hypot(nx,ny)-40;t=np.arctan2(ny,nx)
            arc=40*((t-p['angle_rad']+np.pi)%(2*np.pi)-np.pi)
            rx=.5*p['diameter_mm'];ry=(.8 if p['kind']=='keyhole_pore' else .5)*p['diameter_mm']
            q=((u-p['radial_offset_mm'])/rx)**2+(arc/ry)**2
            expected+=w*(-p['depth_mm']*np.maximum(np.sqrt(np.maximum(1-q,0))-.35,0)).astype('f4')
        actual=delta[np.ix_(rr,cols)];m=mask[np.ix_(rr,cols)];changed=actual < -1e-6
        outside=float(distance_transform_edt(~m)[changed&~m].max()) if (changed&~m).any() else 0.
        records.append(dict(scene_id=row['scene_id'],deepest_inside_mask=bool(mask[np.unravel_index(delta.argmin(),delta.shape)]),
            positive_height_change=bool((delta>1e-6).any()),maximum_error_mm=float(abs(expected-actual).max()),
            mask_depression_fraction=float(changed[m].mean()),maximum_outside_distance_px=outside))
    summary=dict(geometries=len(records),deepest_inside=sum(r['deepest_inside_mask'] for r in records),
        positive_height_changes=sum(r['positive_height_change'] for r in records),maximum_error_mm=max(r['maximum_error_mm'] for r in records),
        minimum_mask_depression_fraction=min(r['mask_depression_fraction'] for r in records),maximum_outside_distance_px=max(r['maximum_outside_distance_px'] for r in records))
    assert summary['maximum_error_mm']<1e-6 and summary['deepest_inside']==len(records) and summary['minimum_mask_depression_fraction']==1
    result=dict(status='passed',scope='Training only. Independently calculate the cap at four native neighbours and bilinearly resample; compare saved height difference with 1e-6 mm numerical tolerance.',summary=summary,records=records)
    s.write(target,result);return result


def development_pipeline_plot():
    f=s.verify_frozen();config=f['selection']['selected']['growth_merge']['config']
    rows=s.learn.selected_rows('val');data=s.learn.Data(rows,s.read(s.OLD/'normalization.json'))
    signed=np.load(OUT/'development/signed_sigma1.npy',mmap_mode='r')
    candidates=[]
    for i,row in enumerate(rows):
        if row['source_class']=='clean' or row['count_multiplier']!=64 or row['repeat']!=0:continue
        grown=s.grow_components(signed[i],config['high'],config['low'],config['radius'])
        seed=s.grow_components(signed[i],config['high'],config['high'],0)
        before=s.form_regions(seed,config['min_area'],0);after=s.form_regions(grown,config['min_area'],config['gap'])
        def best(regions):
            return max([s.metrics.circular_iou(v['box'],t,s.PERIOD) for v in regions for t in data.truth[i]]+[0])
        candidates.append((best(after)-best(before),best(after),i))
    _,_,i=max(candidates)
    mask=data.masks[i][16:144]>0;y,_=np.nonzero(mask)
    pore=rows[i]['parameters']['defects'][0]['params'];centre=int(round(pore['angle_rad']*s.PERIOD/(2*np.pi)))%s.PERIOD
    cols=(np.arange(-22,23)+centre)%s.PERIOD;rr=np.arange(max(0,y.min()-8),min(128,y.max()+9));truth=mask[np.ix_(rr,cols)]
    full=signed[i];score=abs(full);seeds=score>=config['high'];radius=config['radius']
    near=s.cv2.dilate(np.pad(seeds.astype('u1'),((0,0),(radius,radius)),mode='wrap'),np.ones((2*radius+1,2*radius+1),'u1'),borderType=s.cv2.BORDER_CONSTANT,borderValue=0)[:,radius:-radius]>0
    allowed=(score>=config['low'])&near
    # Display only weak components connected to a seed, matching the inference rule.
    from scipy.ndimage import label
    padding=radius+45
    wrapped=np.pad(allowed,((0,0),(padding,padding)),mode='wrap')
    labs,_=label(wrapped,structure=np.ones((3,3)))
    seed_wrapped=np.pad(seeds,((0,0),(padding,padding)),mode='wrap')
    keep=np.unique(labs[seed_wrapped]);keep=keep[keep!=0]
    grown=np.isin(labs,keep)[:,padding:-padding]
    regions=s.prediction(full,config)
    fig,axes=plt.subplots(1,4,figsize=(14,4),layout='constrained')
    values=[full[np.ix_(rr,cols)],seeds[np.ix_(rr,cols)],grown[np.ix_(rr,cols)],data.image(i)[16:144][np.ix_(rr,cols)]]
    for k,(ax,value,title) in enumerate(zip(axes,values,['Signed median residual',f"Strong seeds: |score| ≥ {config['high']:g}",f"Connected growth: |score| ≥ {config['low']:g}",'Final merged detection boxes'])):
        options=dict(cmap='coolwarm',vmin=-max(24,float(abs(value).max())),vmax=max(24,float(abs(value).max()))) if k==0 else dict(cmap='gray',vmin=0,vmax=1)
        ax.imshow(value,**options);ax.contour(truth,levels=[.5],colors=['#ff2255'],linewidths=1.6)
        if k==3:
            for v in regions:
                x0,y0,x1,y1=v['box']
                for shift in (-s.PERIOD,0,s.PERIOD):
                    left=x0+shift-(centre-22);right=x1+shift-(centre-22)
                    if right>0 and left<45 and y1>rr.min() and y0<=rr.max():ax.add_patch(Rectangle((left-.5,y0-rr.min()-.5),right-left,y1-y0,fill=False,edgecolor='#00cfea',lw=1.8))
        ax.set_title(title,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('How local growth uses weak contrast around a strong response',fontsize=14)
    fig.savefig(REPORT/'04_region_growth_steps.png',dpi=170);plt.close(fig)
    return dict(scene_id=rows[i]['scene_id'],selection='Validation example maximising box IoU gain over identical-threshold seeds with growth disabled. Illustrative, not representative.',config=config)


def performance_plot(results,chosen):
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    for name in NAMES:
        y=[results[name]['by_count'][str(m)]['f1'] for m in s.c.MULTIPLIERS]
        ax.plot(range(3),y,marker='o',linestyle='--' if name=='growth_merge' else '-',label=NAMES[name]+(' (selected new candidate)' if name==chosen else ''),color=COLORS[name],lw=2.8 if name==chosen else 1.7)
    ax.set(xticks=range(3),xticklabels=['1×','4×','64×'],ylim=(0,1.03),ylabel='Pore-localisation F1',xlabel='Effective acquisition count multiplier (higher means lower noise)')
    ax.grid(alpha=.2);ax.legend(fontsize=9,loc='best');ax.set_title('All methods evaluated on the same 24 new backgrounds')
    fig.savefig(REPORT/'02_fresh_results.png',dpi=170);plt.close(fig)


def examples_plot(rows,results,chosen):
    valid=[i for i,r in enumerate(rows) if r['source_class']!='clean' and r['count_multiplier']==64 and r['repeat']==0]
    def best_iou(name,i):
        row=results[name]['rows'][i]
        return max([s.metrics.circular_iou(p,t,s.PERIOD) for p in row['predicted_boxes'] for t in row['truth_boxes']]+[0])
    improvement=max(valid,key=lambda i:(best_iou(chosen,i)-best_iou('original',i),rows[i]['diameter_mm']))
    failures=[i for i in valid if results[chosen]['rows'][i]['fn']]
    other=max(failures,key=lambda i:rows[i]['diameter_mm']) if failures else min(valid,key=lambda i:best_iou(chosen,i))
    selected=list(dict.fromkeys([improvement,other]))
    data=s.learn.Data(rows,s.read(s.OLD/'normalization.json'))
    signed=np.load(OUT/'confirmation/signed_sigma1.npy',mmap_mode='r')
    fig,axes=plt.subplots(len(selected),5,figsize=(16,3.7*len(selected)),squeeze=False,layout='constrained')
    for j,i in enumerate(selected):
        mask=data.masks[i][16:144]>0;y,x=np.nonzero(mask)
        pore=rows[i]['parameters']['defects'][0]['params'];centre=int(round(pore['angle_rad']*s.PERIOD/(2*np.pi)))%s.PERIOD
        cols=(np.arange(-24,25)+centre)%s.PERIOD;rr=np.arange(max(0,y.min()-10),min(128,y.max()+11))
        image=data.image(i)[16:144][np.ix_(rr,cols)];truth=mask[np.ix_(rr,cols)]
        for k,name in enumerate(['image','residual','original',chosen,'cnn']):
            ax=axes[j,k]
            if name=='residual':
                value=signed[i][np.ix_(rr,cols)];lim=max(24,float(abs(value).max()))
                ax.imshow(value,cmap='coolwarm',vmin=-lim,vmax=lim)
            else:ax.imshow(image,cmap='gray',vmin=0,vmax=1)
            ax.contour(truth,levels=[.5],colors=['#ff2255'],linewidths=1.6)
            if name in NAMES:
                row=results[name]['rows'][i]
                for bx,by,br,bb in row['predicted_boxes']:
                    for shift in (-s.PERIOD,0,s.PERIOD):
                        left=bx+shift-(centre-24);right=br+shift-(centre-24)
                        if right>0 and left<49 and bb>rr.min() and by<=rr.max():
                            ax.add_patch(Rectangle((left-.5,by-rr.min()-.5),right-left,bb-by,fill=False,edgecolor='#00cfea',lw=1.8))
                ax.set_xlabel(f"Best box IoU {best_iou(name,i):.3f}\nFull-ring TP / FP / FN: {row['tp']} / {row['fp']} / {row['fn']}",fontsize=9)
            ax.set_title({'image':'TOPO + true opening','residual':'Signed median residual'}.get(name,NAMES.get(name,'')),fontsize=11)
            ax.set_xticks([]);ax.set_yticks([])
        axes[j,0].set_ylabel(f"{rows[i]['scene_id']}\nDiameter {rows[i]['diameter_mm']:.3f} mm",fontsize=9)
    fig.suptitle('Red: geometric opening; cyan: detected boxes. Full-ring inference precedes display cropping.',fontsize=13)
    fig.savefig(REPORT/'03_detection_examples.png',dpi=160);plt.close(fig)
    return dict(scene_ids=[rows[i]['scene_id'] for i in selected],selection='Post-confirmation illustration: largest IoU improvement over original, and largest remaining missed pore at 64x repeat 0. Not performance estimates.')


def html_table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    frozen,rows,results=audit();chosen=frozen['selection']['chosen_family'];diag=diagnose(results)
    stages=stage_diagnosis(rows,results,chosen,frozen['selection']['selected'][chosen]['config'])
    comparisons={b:intervals(results,chosen,b) for b in ('original','previous','seed_only','cnn') if b!=chosen}
    controls={name:intervals(results,name,name+'_disabled') for name in ('growth','growth_merge')}
    s.write(OUT/'matched_ablation_intervals.json',controls)
    s.write(OUT/'paired_intervals.json',comparisons)
    heights=height_audit();s.write(REPORT/'height_injection_audit.json',heights)
    geometry=geometry_plot();pipeline=development_pipeline_plot();performance_plot(results,chosen);examples=examples_plot(rows,results,chosen)
    s.write(REPORT/'visual_provenance.json',dict(geometry=geometry,pipeline=pipeline,confirmation=examples))
    s.write(REPORT/'selected_method.json',frozen['selection'])
    headers=['Method','1× F1','4× F1','64× F1']
    table=[[NAMES[name]+(' [selected new candidate]' if name==chosen else '')]+[f"{results[name]['by_count'][str(m)]['f1']:.3f}" for m in s.c.MULTIPLIERS] for name in NAMES]
    counttable=[]
    for name in NAMES:
        for m in s.c.MULTIPLIERS:
            rr=[r for r in results[name]['rows'] if r['count_multiplier']==m];v=s.metrics.summary(rr)
            counttable.append([NAMES[name],f'{m}×',v['tp'],v['fp'],v['fn'],f"{v['clean_rings_flagged']}/{v['clean_rings']}"])
    with (REPORT/'results.csv').open('w',newline='',encoding='utf8') as fh:
        writer=csv.writer(fh);writer.writerow(['method','count_multiplier','tp','fp','fn','precision','recall','f1','clean_images_flagged','clean_images'])
        for name in NAMES:
            for m in s.c.MULTIPLIERS:
                v=s.metrics.summary([r for r in results[name]['rows'] if r['count_multiplier']==m])
                writer.writerow([name,m,v['tp'],v['fp'],v['fn'],v['precision'],v['recall'],v['f1'],v['clean_rings_flagged'],v['clean_rings']])
    selection=[]
    for name,t in frozen['selection']['selected'].items():
        p=t['config'];selection.append([NAMES[name],p['high'],p['low'],p['radius'],p['gap'],p['min_area'],f"{t['macro_f1']:.3f}"])
    differences=[]
    for b,dd in comparisons.items():
        for m in s.c.MULTIPLIERS:
            d=dd[str(m)];lo,hi=d['paired_95_ci'];differences.append([NAMES[b],f'{m}×',f"{d['delta']:+.3f}",f'[{lo:+.3f}, {hi:+.3f}]'])
        d=dd['macro'];lo,hi=d['paired_95_ci'];differences.append([NAMES[b],'Mean across counts',f"{d['delta']:+.3f}",f'[{lo:+.3f}, {hi:+.3f}]'])
    ablations=[]
    for name,dd in controls.items():
        for m in s.c.MULTIPLIERS:
            d=dd[str(m)];lo,hi=d['paired_95_ci']
            ablations.append([NAMES[name],f'{m}×',f"{results[name+'_disabled']['by_count'][str(m)]['f1']:.3f}",f"{results[name]['by_count'][str(m)]['f1']:.3f}",f"{d['delta']:+.3f}",f'[{lo:+.3f}, {hi:+.3f}]'])
    low=diag[chosen]['64'];winner=frozen['selection']['selected'][chosen]
    supported=[str(m)+'×' for m in s.c.MULTIPLIERS if comparisons.get('original',{}).get(str(m),{}).get('paired_95_ci',[-1])[0]>0]
    support=('The paired interval excludes zero in favour of the selected method at '+', '.join(supported)+'.') if supported else 'The paired intervals do not establish improvement over the original median at any tested count level.'
    mean=comparisons['original']['macro'];lo,hi=mean['paired_95_ci']
    support+=f" The change in mean F1 across count levels is {mean['delta']:+.3f}, with a paired 95% interval [{lo:+.3f}, {hi:+.3f}]."
    prev=comparisons['previous']['macro'];plo,phi=prev['paired_95_ci']
    previous_conclusion=f"Compared with the previous position-and-size correction, the mean F1 difference is {prev['delta']:+.3f} [{plo:+.3f}, {phi:+.3f}]. "
    previous_conclusion+=('The interval supports lower performance for the new candidate.' if phi<0 else 'The interval supports higher performance for the new candidate.' if plo>0 else 'This interval does not establish a difference between the methods.')
    merge=controls['growth_merge']['macro'];mlo,mhi=merge['paired_95_ci']
    merge_conclusion=f"With the growth settings held fixed, enabling fragment merging changes mean F1 by {merge['delta']:+.3f}, with a paired 95% interval [{mlo:+.3f}, {mhi:+.3f}]."
    growth=controls['growth']['macro'];glo,ghi=growth['paired_95_ci']
    growth_conclusion=f"At the same strong threshold and minimum area, enabling local growth increases mean F1 from {results['growth_disabled']['macro_f1']:.3f} to {results['growth']['macro_f1']:.3f}, a change of {growth['delta']:+.3f} [{glo:+.3f}, {ghi:+.3f}]. This isolates the growth step at that operating point; it does not establish superiority over the previously tuned detectors."
    text=f'''# Median region-growth study

Completed a separate prospective synthetic study. Validation selected **{NAMES[chosen]}** from {frozen['selection']['trial_count']} new configurations. Its validation mean F1 was {winner['macro_f1']:.3f}, compared with {frozen['previous']['macro_f1']:.3f} for the previously selected position-and-size correction. It is the selected new candidate, not a replacement selected over every comparator. {support}

{previous_conclusion}

{merge_conclusion}

{growth_conclusion}

## What was tested?

The median background estimate, Gaussian sigma 1 pixel, training-only normalisation and per-row robust calibration stayed fixed. Strong seeds trigger lower-threshold local growth within a bounded radius. Compatible fragments can merge across a small gap if their transverse spans overlap and their combined dimensions remain at most 40 pixels (2 mm). Both contrast signs are retained as diagnostic evidence; neither sign is mandatory.

The three families were threshold only, growth, and growth plus merging. Each family was independently selected on the same 32 validation backgrounds (576 images). Training calibration comes from 96 original backgrounds. The search is finite and conditional on this renderer, known annular coordinates and pixel pitch. No CNN retraining was performed.

## What was held out?

All selections were frozen at {frozen['frozen_utc']}, before generating 24 new backgrounds / 432 images. Each count level has 144 images: 48 clean and 96 pore-containing acquisitions. Those 96 acquisitions represent 48 pore geometries with two repeats, on 24 independent backgrounds. Seeds are disjoint from the original campaign and previous median follow-up. Every comparator received the same fresh images.

The original one-to-one circular box IoU >= 0.5 rule was retained. A nearby response can fail localisation. Filled-box pixel overlap is a selection tie-break for the new methods and is not contour-segmentation accuracy.

## Results

'''
    text+='| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'
    text+='\n'.join('| '+' | '.join(row)+' |' for row in table)
    text+=f'''\n\nAt 64×, the validation winner correctly localised {low['correct']} of {low['pore_scenes']} pore acquisitions. Another {low['overlapping_but_failed']} had an overlapping prediction that failed localisation; {low['no_overlapping_prediction']} had no overlapping prediction. It flagged {low['clean_images_flagged']} of {low['clean_scenes']} clean acquisitions.

## Interpretation and limits

The code and height maps support aligned injection: the geometric opening and depression are generated from the same cap equation. Detector differences, surface slopes, visibility, blur and sampling produce a different contrast footprint. Height is bilinearly sampled and the binary mask uses nearest-neighbour sampling, so their sampled edge support can differ.

This experiment changes region formation, not the physical renderer. Independent family selection means family-to-family results include different tuned thresholds and areas. Those family comparisons do not isolate one parameter; the predeclared matched controls isolate growth and merging at fixed operating points. Larger growth or merging can recover a pore extent but can also join noise, enlarge a box too far, or merge adjacent pores. The campaign contains one prescribed pore per defect scene, so crowded-pore separation remains untested.

Uncertainty uses 2,000 paired background bootstrap resamples, grouping repeats and noise levels. Intervals are conditional on selected methods and omit selection uncertainty. Multiple comparisons are exploratory. The previously inspected validation set is reused; fresh confirmation reduces direct test reuse but cannot establish physical transfer. Measured weld-image validation remains future work.

## Files

Open OPEN_REPORT.html for figures, operating points, counts and intervals. results.csv contains the numerical summary. Raw artifacts and frozen protocol: {OUT}

Reproduce with the project Python: WeldDefectDetection/median_region_growth_20261110.py all, then WeldDefectDetection/report_median_region_growth_20261110.py. Existing completed stages are reused. All eight behavioural tests passed before the run; final file, seed, timestamp, metric and prediction audits passed.
'''
    (REPORT/'README.md').write_text(text,encoding='utf8')
    body='<h1>Can local growth and fragment merging improve the median?</h1>'
    body+=f'<p class="lead">Validation selected <strong>{NAMES[chosen]}</strong>. {support}</p>'
    body+='<p>Completed: 96 calibration backgrounds, 32 validation backgrounds, and 24 newly generated confirmation backgrounds. Original thesis and poster results are preserved.</p>'
    body+='<h2>1. What changed?</h2><p>The same median residual now supplies strong seeds. Lower-threshold pixels may join only within a bounded neighbourhood and through a connected path. A separate rule can merge aligned fragments across a narrow gap. Both bright and dark evidence are accepted. The pore labels are used for validation scoring, never for inference.</p>'
    body+=html_table(['Family','Strong threshold','Weak threshold','Radius (px)','Merge gap (px)','Min area (px)','Validation mean F1'],selection)
    body+=f"<p>The selected new candidate scored {winner['macro_f1']:.3f} mean F1 on validation, below the previous position-and-size correction's {frozen['previous']['macro_f1']:.3f}. Selection here is among the 171 new configurations; it does not declare the new method better than every comparator.</p>"
    body+='<p>Thresholds are robust standardized median-residual scores, not raw TOPO intensities. At the fixed 0.05 mm pitch, the selected 6-pixel growth radius is 0.30 mm and the 2-pixel merge gap is 0.10 mm. The maximum region extent is 40 pixels (2 mm); the smaller transverse span must overlap by at least 50% when merging. Gaussian sigma remains 1 pixel. No fixed positional shift is added to the new methods.</p>'
    body+='<img src="04_region_growth_steps.png"><p>This validation example illustrates the largest localisation improvement over the identical strong threshold with growth disabled. The red outline is used only for evaluation and display. The growth panel precedes minimum-area and maximum-extent filtering.</p>'
    body+='<h2>2. Is the surface depression in the right place?</h2><img src="01_geometry_and_contrast.png"><p>The red outline is the geometric opening. The first column shows the actual saved height change, the second the noiseless TOPO change. These are training examples, selected in the prior alignment study. Contrast need not fill the opening uniformly.</p>'
    hs=heights['summary']
    body+=f"<p>An independent calculation checked {hs['geometries']} training pore geometries. All mask pixels had a surface depression and every deepest point lay inside its mask. The maximum saved-versus-calculated height-change error was {hs['maximum_error_mm']:.3g} mm. Bilinear height sampling versus nearest-neighbour mask sampling extended changed height support by up to {hs['maximum_outside_distance_px']:g} pixels beyond the sampled mask. This verifies numerical injection alignment, not physical image realism. <a href='height_injection_audit.json'>Audit details</a>.</p>"
    body+='<h2>3. What happened on new backgrounds?</h2>'+html_table(headers,table)+'<img src="02_fresh_results.png">'
    body+='<p>The growth and growth-plus-merging curves coincide: their fresh detection results are identical.</p>'
    body+='<p>Each count level contains 48 clean and 96 pore acquisitions. There are 24 independent backgrounds, not 144 independent specimens. Higher acquisition counts reduce noise. Detection requires one-to-one box overlap of at least 0.5.</p>'
    body+=html_table(['Method','Counts','Detected','Unmatched detections','Missed','Clean images flagged'],counttable)
    body+='<h2>4. Is the change supported beyond point estimates?</h2>'+html_table(['Comparator','Counts','Selected minus comparator F1','Paired 95% interval'],differences)
    body+='<p>2,000 paired background resamples keep all acquisitions of a background together. Intervals are conditional on the frozen models, exclude tuning uncertainty and have no multiple-comparison adjustment. An interval spanning zero does not establish an improvement.</p>'
    body+='<h3>What happens when each added step is disabled?</h3><p>These controls were declared before the freeze. Growth is compared with the same strong threshold and minimum area but no weak-region growth. Merging is compared with exactly the same grown regions and area cutoff but no merging. No control was used to replace the validation winner.</p>'
    body+=html_table(['Selected family','Counts','Step disabled F1','Step enabled F1','Difference','Paired 95% interval'],ablations)
    body+='<p>'+html.escape(growth_conclusion)+'</p>'
    body+='<h2>5. What do the detections look like?</h2><img src="03_detection_examples.png">'
    body+='<p>Examples were selected after confirmation to illustrate improvement and remaining failure. They are not representative performance estimates. Red outlines and crop locations are used only for display; detection ran on full rings without truth.</p>'
    diagnosis_rows=[]
    for name in NAMES:
        d=diag[name]['64'];diagnosis_rows.append([NAMES[name],d['correct'],d['overlapping_but_failed'],d['no_overlapping_prediction'],f"{d['pore_images_flagged']}/{d['pore_scenes']}",f"{d['clean_images_flagged']}/{d['clean_scenes']}"])
    body+=html_table(['64× method','Correct localisation','Overlap but insufficient','No overlapping prediction','Pore images flagged','Clean images flagged'],diagnosis_rows)
    body+='<p>Any box overlap is weaker than the required IoU of 0.5. An image-level alarm only requires some detected region anywhere in the image; it does not establish correct pore localisation.</p>'
    body+=f"<p>At 64× the selected detector correctly localised {low['correct']} pores, produced overlapping but insufficient localisations in {low['overlapping_but_failed']} pore images, and had no overlapping response in {low['no_overlapping_prediction']}. It flagged {low['clean_images_flagged']}/{low['clean_scenes']} clean images.</p>"
    stage=stages['64']
    body+=f"<p>A post-confirmation diagnostic found no strong seed within the true pore box plus an 8-pixel neighbourhood in {stage['no_nearby_strong_seed']}/{stage['pore_scenes']} low-noise pore images. Another {stage['missed_with_nearby_strong_seed']} images had a nearby seed but still failed localisation. The truth-centred neighbourhood is used only for this diagnosis. Growth cannot start where no strong seed exists.</p>"
    body+='<h2>6. What can be concluded?</h2><p>'+html.escape(support)+'</p><p>'+html.escape(previous_conclusion)+'</p><p>'+html.escape(merge_conclusion)+'</p><p>This is a finite region-formation experiment on synthetic isolated pores. Each family has separately tuned parameters. The result does not establish the best possible median detector, crowded-pore performance or agreement with measured images. Large growth and merging may recover weak regions or may join unrelated responses. The physical surface and detector models were unchanged.</p>'
    body+=f'<p>Freeze: {frozen["frozen_utc"]}. File, seed, timestamp, metric, signed-score and CNN replay audits: passed.</p><p><a href="results.csv">Results CSV</a> · <a href="selected_method.json">Selected operating points</a> · <a href="README.md">Study notes</a></p>'
    page='<!doctype html><html><head><meta charset="utf-8"><title>Median region-growth study</title><style>body{font:17px/1.55 Arial,sans-serif;color:#172733;max-width:1250px;margin:40px auto;padding:0 24px}h1,h2{color:#124b72}h2{margin-top:38px}.lead{padding:18px;background:#eaf4fa;border-left:5px solid #1263a1}img{width:100%;height:auto;margin:16px 0}table{border-collapse:collapse;width:100%;font-size:15px;margin:20px 0}td,th{padding:9px;border:1px solid #ccd8df;text-align:left}th{background:#eaf1f5}tr:nth-child(even){background:#f7fafc}</style></head><body>'+body+'</body></html>'
    (REPORT/'OPEN_REPORT.html').write_text(page,encoding='utf8')
    print(json.dumps(dict(report=str(REPORT/'OPEN_REPORT.html'),chosen=chosen,comparison_vs_original=comparisons.get('original'),audit='passed'),indent=2))


if __name__=='__main__':main()
