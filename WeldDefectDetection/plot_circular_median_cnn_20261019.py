"""Matched full-circle and local CV comparison from verified saved data."""
from pathlib import Path
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'WeldDefectDetection'),str(ROOT/'DataSimulation')]
from raised_acquisition import campaign as campaign
from raised_crown import learning as base, rendering
from circular_study import metrics


def main():
    saved=ROOT/'docs/figures/clean_pore_walkthrough_20261012'
    native=ROOT/'docs/figures/full_ring_examples_20261012'
    provenance=campaign.read(saved/'provenance.json')
    assert campaign.sha(campaign.OUT/'frozen_models.json')==provenance['model_manifest_sha256']
    frozen=campaign.read(campaign.OUT/'frozen_models.json')
    norm=campaign.read(campaign.OUT/'normalization.json')
    x,y=provenance['viewport']['x'],provenance['viewport']['y']
    specs={'median':frozen['median'],'coverage':frozen['arms']['coverage']['cnn'],'cnn':frozen['arms']['hard_negative']['cnn']}
    out=ROOT/'docs/figures/circular_median_cnn_20261019';out.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':10})
    fig,axes=plt.subplots(2,5,figsize=(20,8.7))
    fig.subplots_adjust(left=.045,right=.995,bottom=.18,top=.83,wspace=.23,hspace=.48)
    colours={'truth':'#27d97d','median':'#ffb242','coverage':'#10cef1','cnn':'#10cef1'}
    records={};sourcehashes={}
    for row,kind in enumerate(('clean','open_intrinsic')):
        record=provenance['cases'][kind];source=Path(record['source'])
        assert campaign.sha(source)==record['source_sha256']
        sourcehashes[str(source)]=campaign.sha(source)
        with np.load(source) as z:image=base.normalize(z['raw'][0],norm);truth=z['mask'][16:144];raw=z['raw'][0]
        native_path=native/(kind+'_native.npz');prediction_path=saved/(kind+'_predictions.npz')
        sourcehashes[str(native_path)]=campaign.sha(native_path);sourcehashes[str(prediction_path)]=campaign.sha(prediction_path)
        with np.load(native_path) as z:
            topo=z['topo'];valid=z['valid'];native_truth=z['mask']
        assert np.array_equal(rendering.extract(topo),raw)
        assert np.array_equal(rendering.extract(native_truth,order=0)[16:144].astype('u1'),truth)
        with np.load(prediction_path) as z:predictions={k:z[k] for k in z.files}
        rows={}
        for method in specs:
            op=specs[method]['selection']['best']
            mask,boxes=metrics.binary(predictions[method+'_score'],op['threshold'],op['min_area'],op['closing'])
            assert np.array_equal(mask,predictions[method+'_mask'])
            rows[method]=metrics.match(boxes,metrics.regions(truth)[1],5024)
            assert rows[method]==record['whole_ring_metrics'][method]
        records[kind]=rows
        ax=axes[row,0]
        cmap=plt.colormaps['gray'].copy();cmap.set_bad('#ecf0f4')
        ax.imshow(np.ma.array(base.normalize(topo,norm),mask=~valid),origin='lower',cmap=cmap,vmin=0,vmax=1,extent=(-48.025,48.025,-48.025,48.025))
        theta=2*np.pi*np.array([x-32,x+32])/5024
        radii=40+(np.array([y-32,y+32])-79.5)*.05
        angles=np.linspace(theta[0],theta[1],40)
        px=np.r_[radii[0]*np.cos(angles),radii[1]*np.cos(angles[::-1]),radii[0]*np.cos(angles[:1])]
        py=np.r_[radii[0]*np.sin(angles),radii[1]*np.sin(angles[::-1]),radii[0]*np.sin(angles[:1])]
        ax.plot(px,py,color='#d64ec8',lw=1.8)
        centre=np.mean(radii)*np.array([np.cos(theta.mean()),np.sin(theta.mean())])
        ax.annotate('Enlarged region',xy=centre,xytext=(3,-25),color='#773470',fontsize=10,
                    arrowprops=dict(arrowstyle='->',color='#b638a8',lw=1.2))
        ax.set_xlim(-47,47);ax.set_ylim(-47,47);ax.set_xticks([-40,0,40]);ax.set_yticks([-40,0,40])
        ax.set_xlabel('x (mm)');ax.set_ylabel(('CLEAN' if row==0 else 'INJECTED PORE')+'\ny (mm)',fontweight='bold')
        ax.set_title('Full circular TOPO\n80 mm weld centreline diameter')
        bg=base.tile(image,y,x);crop_truth=base.tile(truth,y-16,x)
        extent=(-1.6,1.6,(y-32-79.5)*.05,(y+32-79.5)*.05)
        xvals=(np.arange(64)-31.5)*.05;yvals=(np.arange(64)+y-32-79.5)*.05
        titles=['TOPO + true defect boundary','Neighbouring median','Coverage CNN','Hard-negative CNN']
        for col in range(1,5):
            ax=axes[row,col];ax.imshow(bg,origin='lower',cmap='gray',vmin=0,vmax=1,extent=extent,interpolation='nearest')
            if crop_truth.any():ax.contour(xvals,yvals,crop_truth,[.5],colors=[colours['truth']],linestyles='--',linewidths=1.7)
            title=titles[col-1]
            if col>=2:
                method=('median','coverage','cnn')[col-2]
                crop=base.tile(predictions[method+'_mask'],y-16,x)
                if crop.any():ax.contour(xvals,yvals,crop,[.5],colors=[colours[method]],linewidths=1.8)
                counts=rows[method]
                title+=f'\n{counts["tp"]} correct / {counts["fp"]} false / {counts["fn"]} missed'
            else:title+='\nNo injected defect' if row==0 else '\nSurface-opening pore, diameter 0.88 mm'
            ax.set_title(title);ax.set_xlabel('Local arc offset (mm)');ax.set_ylabel('Radial offset (mm)')
    fig.suptitle('Circular weld: matched clean and defect-injected images\nMedian baseline and both CNNs, using the same TOPO input',fontsize=19,fontweight='bold',y=.98)
    fig.legend([Line2D([0],[0],color=colours['truth'],ls='--',lw=2),Line2D([0],[0],color=colours['median'],lw=2),Line2D([0],[0],color=colours['cnn'],lw=2)],
        ['True geometric defect boundary','Median prediction','CNN prediction'],loc='lower center',bbox_to_anchor=(.5,.055),ncol=3,frameon=False,fontsize=12)
    fig.text(.5,.022,'Historical example test-008, raised_acquisition_20261006_v1. Shared intensity scale. Counts are for whole rings at box IoU >= 0.50.\nEnlarged view selected for illustration after label-free inference. Light-grey area outside the rendered annulus is padding. Synthetic evidence.',ha='center',fontsize=10)
    for suffix in ('png','pdf'):fig.savefig(out/f'circular_clean_pore_comparison.{suffix}',dpi=210)
    plt.close(fig)
    campaign.write(out/'provenance.json',dict(campaign=campaign.OUT.name,case='test-008',source_hashes=sourcehashes,
        model_manifest_sha256=provenance['model_manifest_sha256'],viewport=provenance['viewport'],whole_ring_metrics=records,
        original_native_extract_identical=True,saved_masks_reproduced_from_scores=True,shared_scale=norm,
        primary_matching_iou=.5,physical_validation=False))
    print(out/'circular_clean_pore_comparison.png',flush=True)


if __name__=='__main__':main()
