"""Chapter 4 figures: frozen methods and first training example, no tuning."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'DataSimulation'),str(ROOT/'WeldDefectDetection')]
from count_study import campaign as c, learning as learn
from raised_crown import learning as base
from circular_study import metrics
from src.algorithms.classical.cnn_detector import TinyDefectSegNet


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    c.verify();learn.verify_development()
    figdir=HERE/'figures';evidence=HERE/'evidence'
    figdir.mkdir(exist_ok=True);evidence.mkdir(exist_ok=True)
    frozen=c.read(c.OUT/'frozen_models.json');norm=c.read(c.OUT/'normalization.json')
    method=frozen['arms']['mixed_96']['median']
    rows=[r for r in c.read(c.OUT/'datasets/train/000/index.json')
          if r['count_multiplier']==1 and r['repeat']==0 and r['source_class'] in ('clean','open_intrinsic')]
    hashes={}
    for row in rows:
        for path_key,hash_key in [('raw_path','raw_sha256'),('geometry_path','geometry_sha256')]:
            p=Path(row[path_key]);assert sha(p)==row[hash_key];hashes[str(p.relative_to(ROOT))]=sha(p)
    for p in [c.OUT/'frozen_models.json',c.OUT/'normalization.json',c.OUT/'development_protocol.json']:
        hashes[str(p.relative_to(ROOT))]=sha(p)
    split_counts={}
    for split,n in c.COUNTS.items():
        path=c.OUT/f'datasets/{split}/index.json';allrows=c.read(path)
        hashes[str(path.relative_to(ROOT))]=sha(path)
        examples=[r for r in allrows if r['source_class']=='open_intrinsic' and r['count_multiplier']==1 and r['repeat']==0]
        assert len(allrows)==n*18 and len(examples)==n
        split_counts[split]=dict(backgrounds=n,scenes=len(allrows),nominal_size_bins=[sum(lo<=r['diameter_mm']<hi for r in examples) for lo,hi in [(.15,.30),(.30,.60),(.60,1.01)]])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    def save(fig,name):
        for ext in ['png','pdf','svg']:fig.savefig(figdir/f'{name}.{ext}',dpi=200,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    def box(ax,x,y,w,h,text,color='#e5f0f7',size=10):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015',fc=color,ec='#4b6175',lw=1.1))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)
    def arrow(ax,a,b,color='#4b6175',style='-'):
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=color,lw=1.4,linestyle=style))

    fig,ax=plt.subplots(figsize=(12,6));ax.set(xlim=(0,12),ylim=(0,6));ax.axis('off')
    box(ax,3.4,5.0,5.2,.65,'176 independent backgrounds\nSplit identity precedes all related variants',size=12)
    for x,title,n,scenes,color in [(0.3,'Training',96,1728,'#e1f2e9'),(4.3,'Validation',32,576,'#e4eef9'),(8.3,'Confirmation',48,864,'#f8eadb')]:
        arrow(ax,(6,5),(x+1.7,4.45))
        box(ax,x,3.65,3.4,.8,f'{title}\n{n} backgrounds; {scenes:,} scenes',color,12)
        box(ax,x,2.25,3.4,.9,'Per background: 3 surface conditions\n× 3 count levels × 2 repeats\n= 18 related scenes',color)
        arrow(ax,(x+1.7,3.65),(x+1.7,3.15))
    box(ax,.3,.65,3.4,1.05,'Fit normalisation and median scale\nTrain CNN weights\nNested 24 / full 96 backgrounds','#e1f2e9')
    box(ax,4.3,.65,3.4,1.05,'Choose scale / checkpoint / seed\nand region operating point\nFreeze all choices','#e4eef9')
    box(ax,8.3,.65,3.4,1.05,'Generate after the freeze\nScore complete rings\nNo model or threshold selection','#f8eadb')
    for x in [.3,4.3,8.3]:arrow(ax,(x+1.7,2.25),(x+1.7,1.7))
    ax.text(6,.15,'All variants of one background remain in one split; scenes and patches do not add independent backgrounds.',ha='center',fontsize=10)
    save(fig,'figure_4_1_grouped_design')

    data=learn.Data(rows,norm);op=method['selection']['best'];spec=method['spec']
    cols=np.arange(-32,33);idx=cols%5024;rad=slice(48,112)
    extent=[(cols[0]-.5)*2*np.pi*40/5024,(cols[-1]+.5)*2*np.pi*40/5024,-1.6,1.6]
    fig,axes=plt.subplots(2,4,figsize=(13.5,7.2),layout='constrained')
    stats=[]
    for rownum,row in enumerate(rows):
        x=data.image(rownum,spec['sigma']);residual=learn.median_residual(x);reference=x-residual
        score=learn.median_score(x,spec);mask,boxes=metrics.binary(score,op['threshold'],op['min_area'],op['closing'])
        truth=data.masks[rownum][rad][:,idx]
        arrays=[x[rad][:,idx],reference[rad][:,idx],score[32:96][:,idx],mask[32:96][:,idx]]
        for j,a in enumerate(arrays):
            ax=axes[rownum,j];im=ax.imshow(a,origin='lower',extent=extent,interpolation='nearest',cmap='magma' if j==2 else 'gray',vmin=0,vmax=24 if j==2 else 1)
            if j in (2,3) and truth.any():ax.contour(truth,levels=[.5],origin='lower',extent=extent,colors=['#30bd76'],linestyles='dashed',linewidths=1.2)
            if rownum==0:ax.set_title(['Smoothed, normalised input','Eight-neighbour reference','Standardised residual score','Final predicted mask'][j])
            if rownum==1:ax.set_xlabel('Nominal arc from pore centre (mm)')
            if j==0:ax.set_ylabel(('Clean' if row['source_class']=='clean' else 'Intrinsic pore')+'\nCross-weld u (mm)')
            if rownum==1:fig.colorbar(im,ax=axes[:,j],shrink=.65,pad=.02)
        stats.append(dict(scene_id=row['scene_id'],full_ring_predicted_regions=len(boxes),crop_positive_pixels=int(arrays[-1].sum())))
    fig.suptitle(f'Frozen mixed-96 median on train-000 at 1× counts\nσ = {spec["sigma"]:g} px; threshold = {op["threshold"]:g}; closing {op["closing"]}×{op["closing"]}; minimum area {op["min_area"]} px\nGreen dashed contour: geometry truth; score display clips at 24',fontsize=12)
    save(fig,'figure_4_2_median_steps')

    model=TinyDefectSegNet(in_channels=1,base_channels=24).eval();shapes={}
    handles=[]
    for name in ['enc1','pool1','enc2','pool2','bottleneck','up2','dec2','up1','dec1','head']:
        handles.append(getattr(model,name).register_forward_hook(lambda m,a,b,name=name:shapes.__setitem__(name,list(b.shape))))
    with torch.inference_mode():out=model(torch.zeros((1,1,64,64)))
    for h in handles:h.remove()
    assert out.shape==(1,1,64,64)
    count=sum(p.numel() for p in model.parameters());assert count==262777
    fig,ax=plt.subplots(figsize=(13,6.5));ax.set(xlim=(0,13),ylim=(0,6.5));ax.axis('off')
    box(ax,.2,5.2,2.3,.7,'Input patch\n1 × 64 × 64')
    box(ax,3.2,5.2,2.3,.7,'Encoder 1\n24 × 64 × 64');arrow(ax,(2.5,5.55),(3.2,5.55))
    box(ax,3.2,3.5,2.3,.7,'Encoder 2\n48 × 32 × 32');arrow(ax,(4.35,5.2),(4.35,4.2))
    ax.text(4.55,4.7,'2×2 max pool',fontsize=9)
    box(ax,3.2,1.8,2.3,.7,'Bottleneck\n96 × 16 × 16');arrow(ax,(4.35,3.5),(4.35,2.5))
    ax.text(4.55,3,'2×2 max pool',fontsize=9)
    box(ax,7.3,3.5,2.5,.7,'Decoder 2\n48 × 32 × 32','#e4f1e8');arrow(ax,(5.5,2.15),(8.55,3.5))
    ax.text(7.0,2.05,'Up 96→48\n+ concatenate',ha='center',fontsize=9)
    box(ax,7.3,5.2,2.5,.7,'Decoder 1\n24 × 64 × 64','#e4f1e8');arrow(ax,(8.55,4.2),(8.55,5.2))
    ax.text(8.75,4.7,'Up 48→24\n+ concatenate',fontsize=9)
    arrow(ax,(5.5,5.55),(7.3,5.55),style='--');arrow(ax,(5.5,3.85),(7.3,3.85),style='--')
    ax.text(6.4,5.72,'skip',ha='center',fontsize=9);ax.text(6.4,4.03,'skip',ha='center',fontsize=9)
    box(ax,10.5,5.2,2.3,.7,'1×1 head + sigmoid\n1 × 64 × 64','#f9ecd9');arrow(ax,(9.8,5.55),(10.5,5.55))
    box(ax,10.5,3.5,2.3,.7,'Keep centre\n1 × 32 × 32','#f9ecd9');arrow(ax,(11.65,5.2),(11.65,4.2))
    box(ax,10.5,1.8,2.3,.7,'Tile around ring\n1 × 128 × 5024','#f9ecd9');arrow(ax,(11.65,3.5),(11.65,2.5))
    ax.text(.35,1.05,'Each encoder, bottleneck and decoder block: two 3×3 convolutions, each with BatchNorm and ReLU.\nUp: 2×2 transposed convolution, stride 2. Dashed arrows carry encoder features to the decoder.\nShapes exclude the batch axis: channels × radial rows × angular columns. 262,777 trainable parameters.',fontsize=10,va='center')
    save(fig,'figure_4_3_network')
    receipt=dict(campaign=c.OUT.name,selection='First training background, 1x repeat 0; frozen mixed_96 median',
                 input_hashes=hashes,normalization=norm,median_settings=dict(sigma=spec['sigma'],threshold=op['threshold'],closing=op['closing'],min_area=op['min_area']),
                 example_stats=stats,split_counts=split_counts,network_shapes=shapes,parameters=count,source_hash_checks='generation and development passed',
                 figure_files={p.name:sha(p) for p in sorted(figdir.glob('figure_4_*'))},generator_sha256=sha(__file__),
                 campaign_mutations=False)
    (evidence/'chapter4_provenance.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'parameters':count,'example_stats':stats,'figures':3}))


if __name__=='__main__':main()
