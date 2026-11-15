"""Post-hoc explanatory checks, never used to select a new deployed detector."""
import gc
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from circular_study import generation as g,geometry as geo,evaluation,training,median,metrics
from . import campaign as c

def component(sample,pitch,angle,kind,diameter,seed):
    n=round(6.4/pitch);axis=(np.arange(n)-(n-1)/2)*pitch
    x=40*np.cos(angle)+axis;y=40*np.sin(angle)+axis
    yy,xx=np.meshgrid(y,x,indexing='ij');u=np.hypot(xx,yy)-40
    height=geo.surface(u,np.arctan2(yy,xx),seed=seed,depth=.14,crown=.012,width=1.1,haz=1.,focus=.35,speed=800.,power=18.,amplitude=.1)
    shape=(2,n,n);fusion=abs(u)<=.55
    volume=g.old.VoxelVolumeState(material_z=np.repeat(np.where(fusion,sample.material.z+.2,sample.material.z)[None],2,axis=0).astype('f4'),
        yield_modifier=np.ones(shape,'f4'),void_fraction=np.zeros(shape,'f4'),relief_mm=np.repeat(height[None],2,axis=0).astype('f4'),
        defect_mask=np.zeros(shape,bool),fusion_mask=np.repeat(fusion[None],2,axis=0),x_mm=x,y_mm=y,z_mm=np.array([-.08,0.]),
        spacing_mm=(pitch,pitch,.08),origin_mm=(float(x[0]),float(y[0]),-.08),metadata=dict(surface_plane_index=1,pixel_size_mm=pitch))
    if kind!='clean':geo.inject(volume,[dict(type=kind,params=dict(diameter_mm=diameter,depth_mm=.25,angle_rad=angle,radial_offset_mm=0.))],np.random.default_rng(seed+100))
    section,_=g.old.se.ScanEngine._top_surface_section(volume,min_visible_area_px=1)
    normals=g.old.se.ScanEngine._normals_from_heightmap(section.relief_mm_map,pitch)
    q=g.old.chunked_quadrants(normals=normals,detector=sample.detector,material=sample.material,material_z_map=section.material_z_map,
        yield_modifier_map=np.clip(section.yield_modifier_map*(1-.22*section.void_fraction_map),.15,1.15),
        openness_map=np.clip(.88*section.void_fraction_map,0,1),b_coeff=.920694853227)
    q=gaussian_filter(q,(.05/pitch,.05/pitch,0));factor=round(.05/pitch)
    q=q.reshape(128,factor,128,factor,4).mean(axis=(1,3))
    occupancy=section.defect_mask.reshape(128,factor,128,factor).mean(axis=(1,3))[32:96,32:96]
    return dict(topo=g.baseline.form_channels(q).topo[32:96,32:96],occupancy=occupancy,
        geometric_area_mm2=float(section.defect_mask.sum()*pitch*pitch))

def resolution():
    folder=c.OUT/'investigation';folder.mkdir(exist_ok=True)
    protocol=dict(diameters_mm=[.35,.5,.8],depth_mm=.25,angles_deg=[0,45],seeds=[2250400000,2250400001],
        pitch_mm=[.05,.025,.0125],reference_mm=.0125,output_pitch_mm=.05,psf_sigma_mm=.05,
        scope='Surface/normals/quadrant component, fixed surface planes, unit gains, no raster noise. 0.35 mm is below the training diameter range. No CV selection.',source_sha256=c.sha(__file__))
    if (folder/'resolution_protocol.json').exists():assert c.read(folder/'resolution_protocol.json')==protocol
    else:c.write(folder/'resolution_protocol.json',protocol)
    cfg=g.old.load_config_bundle(c.ROOT/'DataSimulation/config');sample,_=g.baseline.c.sampled(cfg,2250400100,0)
    rows=[]
    for seed in protocol['seeds']:
        for degree in protocol['angles_deg']:
            clean={p:component(sample,p,np.deg2rad(degree),'clean',.5,seed) for p in protocol['pitch_mm']}
            for diameter in protocol['diameters_mm']:
                for kind in ('keyhole_pore','intrinsic_pore'):
                    tag=f'{seed}_{degree}_{diameter}_{kind}';path=folder/(tag+'.json')
                    if path.exists():rows.extend(c.read(path));continue
                    values={p:component(sample,p,np.deg2rad(degree),kind,diameter,seed) for p in protocol['pitch_mm']}
                    reference=values[.0125]['topo']-clean[.0125]['topo'];batch=[]
                    for pitch,v in values.items():
                        delta=v['topo']-clean[pitch]['topo']
                        batch.append(dict(seed=seed,angle_deg=degree,diameter_mm=diameter,kind=kind,pitch_mm=pitch,
                            relative_signal_error=float(np.linalg.norm(delta-reference)/max(np.linalg.norm(reference),1e-12)),
                            geometric_area_mm2=v['geometric_area_mm2'],reference_area_mm2=values[.0125]['geometric_area_mm2'],
                            common_grid_occupancy_l1=float(abs(v['occupancy']-values[.0125]['occupancy']).mean())))
                    c.write(path,batch);rows.extend(batch)
                    np.savez_compressed(folder/(tag+'.npz'),**{f'delta_{p}':v['topo']-clean[p]['topo'] for p,v in values.items()},**{f'occupancy_{p}':v['occupancy'] for p,v in values.items()})
                    print('resolution',tag,flush=True)
    c.write(folder/'resolution.json',dict(protocol=protocol,rows=rows))

def postprocessing():
    torch.set_num_threads(4);frozen=evaluation.verify_freeze();data=g.load('test');model=training.load_model(frozen['cnn'])
    folder=c.OUT/'investigation';folder.mkdir(exist_ok=True);output={}
    for method,scores in [('cnn',training.predict(data,model,frozen['cnn']['config'])),('median',median.scores(data,frozen['median']['spec']))]:
        settings=[];spec=frozen[method]['selection']['best']
        for area in (1,3,15):
            for closing in (1,3):
                s=dict(spec, min_area=area,closing=closing);rows=metrics.evaluate(data,scores,s,retain=True)
                settings.append(dict(min_area=area,closing=closing,threshold=spec['threshold'],metrics=metrics.summary(rows)))
        detail=[]
        for i,(score,truth,meta) in enumerate(zip(scores,data['mask'],data['meta'])):
            mask,bb=metrics.binary(score,spec['threshold'],spec['min_area'],spec['closing']);_,gt=metrics.regions(truth)
            if not gt:continue
            maximum=max((metrics.circular_iou(a,b,score.shape[1]) for a in bb for b in gt),default=0.)
            detail.append(dict(scene_id=meta['scene_id'],kind=meta['source_class'],label_area=int(truth.sum()),
                best_box_iou=maximum,score_peak=float(score[truth>0].max()),
                mean_score_inside=float(score[truth>0].mean()),threshold_pixels_inside=int(((score>=spec['threshold'])&(truth>0)).sum()),
                final_pixels_inside=int(((mask>0)&(truth>0)).sum())))
        output[method]=dict(settings=settings,per_pore=detail,localization_sensitivity={str(t):sum(r['best_box_iou']>=t for r in detail) for t in (.25,.5,.75)})
    c.write(folder/'postprocessing.json',dict(methods=output,
        interpretation='Post-hoc mechanism diagnosis on already opened v4 test data. No operating-point promotion and no new independent performance claim. Threshold stays frozen; the six area/closing combinations and three IoU criteria are all reported.',
        frozen_models_sha256=c.sha(g.OUT/'frozen_models.json')))
    print('postprocessing complete',flush=True)

if __name__=='__main__':postprocessing();resolution()
