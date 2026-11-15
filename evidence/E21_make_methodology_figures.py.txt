"""Reproduce Chapter 3 figures from frozen source and a fixed training example.

No selection by detector performance; never write into the campaign directory.
The first run replays clean and intrinsic train-000 to recover quiet quadrants.
"""
from pathlib import Path
from dataclasses import replace, asdict
import sys, json, hashlib, gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT/'DataSimulation/data/output/count_diversity_20261027_v2'
FIG = HERE/'figures'
EVIDENCE = HERE/'evidence'
FIG.mkdir(exist_ok=True); EVIDENCE.mkdir(exist_ok=True)
sys.path[:0] = [str(ROOT/'DataSimulation'), str(ROOT/'WeldDefectDetection')]
from count_study import campaign as study
from raised_crown import geometry as geo, rendering as render


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol=study.verify()
    rows=study.read(DATA/'datasets/train/000/index.json')
    chosen={r['source_class']:r for r in rows if r['count_multiplier']==1 and r['repeat']==0}
    input_hashes={}
    for r in rows:
        for field,digest in [('raw_path','raw_sha256'),('geometry_path','geometry_sha256')]:
            path=Path(r[field]);input_hashes[str(path.relative_to(ROOT))]=digestval=sha(path)
            assert digestval==r[digest], path
    cache=EVIDENCE/'chapter3_train000_trace.npz'
    receipt_path=EVIDENCE/'chapter3_trace_provenance.json'
    if not cache.exists():
        row=chosen['clean'];seed=row['seed']
        sample,_=render.oldg.baseline.c.sampled(render.old.load_config_bundle(ROOT/'DataSimulation/config'),seed,0)
        sample=replace(sample,beam=replace(sample.beam,dwell_time_us=row['dwell_time_us']))
        assert asdict(sample)==row['parameters']
        engine=render.Engine(seed,row['amplitude']);checks={};out={}
        for kind in ('clean','open_intrinsic'):
            changed=replace(sample,defects=study.pore_plan(seed,0,kind))
            assert asdict(changed)==chosen[kind]['parameters']
            payload,_=render.render(engine,changed)
            with np.load(chosen[kind]['geometry_path']) as saved:
                for key in ('height','mask','quiet'):
                    np.testing.assert_array_equal(payload[key],saved[key])
            np.testing.assert_array_equal(payload['raw'][0],np.load(chosen[kind]['raw_path']))
            checks[kind]='height, mask, quiet TOPO and 1x repeat-0 TOPO match bit-for-bit'
            out[kind+'_quadrants']=np.stack([render.extract(engine.acquisition_quiet_quadrants[...,i]) for i in range(4)])
            print('Replayed and verified',kind,flush=True)
            del payload;gc.collect()
        np.savez_compressed(cache,**out)
        receipt=dict(campaign=DATA.name,selection='first training background; no scores inspected for selection',
                     background_id='train-000',seed=seed,input_hashes=input_hashes,
                     generation_sources_verified=len(protocol['source_hashes']),checks=checks,
                     trace_sha256=sha(cache),generator_sha256=sha(__file__),
                     replay_changes_to_models_or_campaign=False)
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        del engine;gc.collect()
    receipt=json.loads(receipt_path.read_text())
    assert receipt['input_hashes']==input_hashes and receipt['trace_sha256']==sha(cache)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})

    def save(fig,name):
        for ext in ('png','pdf','svg'):
            fig.savefig(FIG/f'{name}.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
        plt.close(fig)

    fig=plt.figure(figsize=(15,4.6),layout='constrained')
    ax=fig.add_subplot(131)
    u=np.linspace(-2.3,2.3,801)
    nominal=3*np.maximum(1-(u/1.4)**2,0)**3
    ax.fill_between(u,0,nominal,color='#c8dfed');ax.plot(u,nominal,color='#1b6d9e',label='Nominal surface')
    depth=np.linspace(0,7,300);width=geo.fusion_half_width(depth)
    ax.fill_betweenx(-depth,-width,width,color='#dfe6eb')
    ax.plot(width,-depth,color='#546878');ax.plot(-width,-depth,color='#546878')
    ax.scatter([1.4,.6,0],[0,-5,-7],s=22,color='#bc513d',zorder=5)
    ax.axhline(0,color='gray',ls=':',lw=.8)
    ax.set(xlabel='Cross-weld position u (mm)',ylabel='Height z (mm)',title='(a) Prescribed section and fusion outline',ylim=(-7.5,3.6))
    ax.text(-2.15,-3.3,'Solid fusion region\nnot an open trench',fontsize=9)
    ax.legend(loc='upper left',fontsize=8)
    ax=fig.add_subplot(132,projection='3d')
    uu,tt=np.meshgrid(np.linspace(-1.8,1.8,65),np.linspace(0,2*np.pi,241))
    zz=geo.height(uu,tt,chosen['clean']['seed'],chosen['clean']['amplitude'])
    ax.plot_surface((40+uu)*np.cos(tt),(40+uu)*np.sin(tt),zz,cmap='viridis',vmin=0,vmax=3.3,
                    linewidth=0,rcount=150,ccount=45,rasterized=True)
    ax.set(xlabel='x (mm)',ylabel='y (mm)',title='(b) Circular raised surface')
    ax.set_box_aspect((1,1,.35));ax.view_init(elev=32,azim=-65)
    ax.text2D(.08,.0,'Height z in mm; vertical scale exaggerated',transform=ax.transAxes,fontsize=9)
    ax=fig.add_subplot(133)
    theta=np.linspace(0,2*np.pi,801)
    for a,col in [(0.,'#243b53'),(.05,'#2e8b72'),(.1,'#b94d39')]:
        ax.plot(np.degrees(theta),geo.height(0.,theta,chosen['clean']['seed'],a),label=f'a = {a:g}',color=col)
    ax.set(xlabel='Angle around the weld (degrees)',ylabel='Centreline height (mm)',title='(c) Declared variation levels',xlim=(0,360))
    ax.legend(fontsize=9);ax.grid(alpha=.15)
    save(fig,'figure_3_1_profile_geometry')

    # Seam-centred local window uses wrapped columns and physical nominal arc.
    cols=np.arange(-25,26);ix=cols%5024;sl=slice(60,100)
    pitch=2*np.pi*40/5024
    extent=[(cols[0]-.5)*pitch,(cols[-1]+.5)*pitch,(60-80)*.05,(100-80)*.05]
    def crop(a):return a[sl][:,ix]
    saved={}
    for kind in chosen:
        with np.load(chosen[kind]['geometry_path']) as a:saved[kind]={k:a[k] for k in a.files}
    kinds=('clean','open_intrinsic','open_keyhole');names=('Clean','Intrinsic pore','Keyhole pore')
    fig,axes=plt.subplots(3,3,figsize=(11.6,8.4),layout='constrained')
    clean=saved['clean']['height']
    hmin=min(float(crop(saved[k]['height']).min()) for k in kinds)
    hmax=max(float(crop(saved[k]['height']).max()) for k in kinds)
    maxdrop=max(float(-(saved[k]['height']-clean).min()) for k in kinds)
    for r,(kind,name) in enumerate(zip(kinds,names)):
        for c,array in enumerate((crop(saved[kind]['height']),crop(saved[kind]['height']-clean),crop(saved[kind]['mask']))):
            ax=axes[r,c]
            opts=({'cmap':'viridis','vmin':hmin,'vmax':hmax} if c==0 else
                  {'cmap':'Blues_r','vmin':-maxdrop,'vmax':0} if c==1 else {'cmap':'gray','vmin':0,'vmax':1})
            im=ax.imshow(array,origin='lower',extent=extent,interpolation='nearest',aspect='equal',**opts)
            if r==0:ax.set_title(('Surface height (mm)','Height change from clean (mm)','Geometry mask (0 or 1)')[c])
            if c==0:ax.set_ylabel(name+'\nCross-weld u (mm)')
            if r==2:ax.set_xlabel('Nominal arc from pore centre (mm)')
            if r==2:fig.colorbar(im,ax=axes[:,c],shrink=.65,pad=.02)
    fig.suptitle('Fixed training background 000: matched geometry and labels',fontsize=13)
    save(fig,'figure_3_2_pore_geometry')

    with np.load(cache) as a:q=a['open_intrinsic_quadrants']
    fig,axes=plt.subplots(2,4,figsize=(14,6.7),layout='constrained')
    qlim=(min(float(crop(x).min()) for x in q),max(float(crop(x).max()) for x in q))
    for i in range(4):
        im=axes[0,i].imshow(crop(q[i]),origin='lower',extent=extent,cmap='gray',vmin=qlim[0],vmax=qlim[1],interpolation='nearest')
        axes[0,i].set_title('Quiet quadrant '+ 'ABCD'[i])
    fig.colorbar(im,ax=axes[0,:],label='Raw quadrant units',shrink=.65,pad=.02)
    lower=[saved['open_intrinsic']['quiet']]+[np.load(DATA/f'datasets/train/000/open_intrinsic_c{m}_r0.npy') for m in (1,4,64)]
    tlim=(min(float(crop(x).min()) for x in lower),max(float(crop(x).max()) for x in lower))
    for i,arr in enumerate(lower):
        ax=axes[1,i];im=ax.imshow(crop(arr),origin='lower',extent=extent,cmap='gray',vmin=tlim[0],vmax=tlim[1],interpolation='nearest')
        ax.contour(crop(saved['open_intrinsic']['mask']),levels=[.5],origin='lower',extent=extent,colors=['#e95b52'],linewidths=.8)
        ax.set_title(('Quiet TOPO','Acquired TOPO: 1x','Acquired TOPO: 4x','Acquired TOPO: 64x')[i])
        ax.set_xlabel('Nominal arc (mm)')
    fig.colorbar(im,ax=axes[1,:],label='Raw TOPO units',shrink=.65,pad=.02)
    for ax in axes[:,0]:ax.set_ylabel('Cross-weld u (mm)')
    fig.suptitle('Intrinsic pore: quadrant response, TOPO and count variation\nRed contours show geometry truth; shared intensity scales within each row',fontsize=12)
    save(fig,'figure_3_3_signal_trace')
    receipt['figure_files']={p.name:sha(p) for p in sorted(FIG.glob('figure_3_*'))}
    receipt['generator_sha256']=sha(__file__)
    receipt['example_parameters']={k:chosen[k]['parameters']['defects'] for k in kinds}
    receipt['analytic_opening_factor']=float(np.sqrt(1-.35**2))
    receipt.pop('example_native_max_depth_mm',None)
    receipt['example_sampled_max_depth_mm']=maxdrop
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print('Created and recorded three Chapter 3 figures.',flush=True)


if __name__=='__main__':main()
