"""Figures from new circular rendering and recorded frozen-model predictions."""
from pathlib import Path
import json
import hashlib
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, default=ROOT/'DataSimulation/data/output/circular_examples_20260928_v1')
parser.add_argument('--output', type=Path, default=ROOT/'docs/figures/circular_weld_20260928')
args = parser.parse_args()
DATA, OUT = args.data, args.output
OUT.mkdir(parents=True, exist_ok=True)
protocol = json.loads((DATA/'protocol.json').read_text())
norm = protocol['normalization']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
cache = {}
def get(name):
    if name not in cache:
        meta = json.loads((DATA/f'{name}.json').read_text())
        assert hashlib.sha256((DATA/f'{name}.npz').read_bytes()).hexdigest() == meta['npz_sha256']
        with np.load(DATA/f'{name}.npz') as a:
            raw, masks, crops, cropmasks = [a[k].copy() for k in ('raw','mask','crops_raw','crops_mask')]
        def normalize(x):
            return np.sqrt(np.clip((x-norm['raw_low'])/(norm['raw_high']-norm['raw_low']),0,1))
        cache[name] = meta, normalize(raw), masks, normalize(crops), cropmasks
    return cache[name]

def full(ax, image):
    ax.imshow(image, cmap='gray', vmin=0, vmax=1, origin='lower', extent=(-48.025,48.025,-48.025,48.025), interpolation='nearest')
    ax.set_xlim(-48,48); ax.set_ylim(-48,48)
    ax.set_xticks([]);ax.set_yticks([])

def polygon(box, sector):
    x0,y0,x1,y1=box
    xs=np.concatenate([np.linspace(x0,x1,12),np.full(12,x1),np.linspace(x1,x0,12),np.full(12,x0)])
    ys=np.concatenate([np.full(12,y0),np.linspace(y0,y1,12),np.full(12,y1),np.linspace(y1,y0,12)])
    angle=np.pi/2+2*np.pi*sector/78+(xs-32)*.05/40
    radius=40+(ys-32)*.05
    return np.stack([radius*np.cos(angle),radius*np.sin(angle)],axis=1)

def full_boxes(ax, rows, truth=False):
    for row in rows:
        for b in row['truth_boxes' if truth else 'boxes_xyxy']:
            ax.add_patch(Polygon(polygon(b,row['sector']),fill=False,
                         edgecolor='#60e8c1' if truth else '#ffad33',lw=1.1))

def counts(meta, method):
    m=meta['metrics'][method]
    return f"Found {m['tp']} | Missed {m['fn']} | Extra {m['fp']}"

fig,axes=plt.subplots(1,3,figsize=(13,5.2))
for ax,level in zip(axes,(0,5,10)):
    meta,image,_,_,_=get(f'a{level:02d}_clean')
    full(ax,image)
    ax.set_title(f'{level}% width/relief variation')
    m=meta['metrics']['cnn']
    ax.set_xlabel(f"CNN: {m['clean_false_alarms']}/78 sectors flagged")
axes[0].plot([-42,-22],[-42,-42],color='white',lw=3)
axes[0].text(-32,-38,'20 mm',color='white',ha='center')
fig.suptitle('80 mm diameter circular weld: clean geometry',fontsize=18,y=.98)
fig.text(.5,.08,'Same background and acquisition parameters. No injected defects.\n'
         'Native rendering: 0.05 mm/pixel. Shared frozen intensity mapping. Images displayed at reduced size.',ha='center',fontsize=10)
fig.subplots_adjust(top=.87,bottom=.21,left=.02,right=.98,wspace=.08)
fig.savefig(OUT/'01_circular_variability.png',dpi=160);plt.close(fig)

kinds=('clean','keyhole_pore','intrinsic_pore')
labels=('Clean weld','Keyhole pore','Intrinsic pore')
fig,axes=plt.subplots(3,4,figsize=(13,11.7))
for i,(kind,label) in enumerate(zip(kinds,labels)):
    meta,image,mask,crops,cropmasks=get('a10_'+kind)
    for ax in axes[i]: full(ax,image)
    axes[i,0].set_ylabel(label,fontsize=12)
    # White marker shows the same sector enlarged in the detail figure.
    axes[i,0].add_patch(Polygon(polygon([0,0,64,64],0),fill=False,edgecolor='white',lw=1.2))
    full_boxes(axes[i,1],meta['predictions']['cnn'],truth=True)
    axes[i,1].set_xlabel('No injected defect' if kind=='clean' else 'Injected defect at top of ring',fontsize=9)
    for j,method in enumerate(('cnn','median'),2):
        full_boxes(axes[i,j],meta['predictions'][method])
        axes[i,j].set_xlabel(counts(meta,method),fontsize=9)
for ax,title in zip(axes[0],('TOPO input','Known geometry','CNN output','Median output')):ax.set_title(title)
fig.suptitle('Circular-weld inspection: detections mapped back to the ring',fontsize=17,y=.98)
fig.text(.5,.945,'10% variation | 78 annular sectors | Green: known defect box | Orange: predicted boxes',ha='center',fontsize=10)
fig.text(.5,.028,'Frozen straight-weld models applied after annular rectification; no retraining or threshold changes.\n'
         'Counts cover the inspected band around the ring. Small defects are enlarged in the next figure.',ha='center',fontsize=10)
fig.subplots_adjust(left=.07,right=.99,top=.89,bottom=.09,wspace=.12,hspace=.24)
fig.savefig(OUT/'02_circular_detection_overview.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(3,4,figsize=(12.8,11.2))
for i,(kind,label) in enumerate(zip(kinds,labels)):
    meta,image,mask,crops,cropmasks=get('a10_'+kind)
    for ax in axes[i]:
        ax.imshow(crops[0],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([])
    axes[i,0].set_ylabel(label,fontsize=12)
    axes[i,0].plot([4,24],[59,59],color='white',lw=3)
    axes[i,0].text(14,55,'1 mm',ha='center',color='white',fontsize=9)
    if cropmasks[0].any():axes[i,1].contour(cropmasks[0],levels=[.5],colors=['#60e8c1'],linewidths=1.6)
    axes[i,1].set_xlabel('No injected defect' if kind=='clean' else 'Injected defect outline',fontsize=10)
    for j,method in enumerate(('cnn','median'),2):
        row=meta['predictions'][method][0]
        for x0,y0,x1,y1 in row['boxes_xyxy']:
            axes[i,j].add_patch(Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,edgecolor='#ffad33',lw=1.6))
        axes[i,j].set_xlabel(f"Found {row['tp']} | Missed {row['fn']} | Extra {row['fp']}",fontsize=10)
for ax,title in zip(axes[0],('Rectified TOPO crop','Known geometry','CNN output','Median output')):ax.set_title(title)
fig.suptitle('Enlarged inspection region at the top of the circular weld',fontsize=17,y=.98)
fig.text(.5,.941,'Same geometric location for all three conditions. These are crops from the rendered circular specimens.',ha='center',fontsize=10)
fig.text(.5,.027,'10% variation | Green: known defect | Orange: prediction | Frozen normalization and decision thresholds\n'
         'Found requires box IoU >= 0.5. A box at the correct location can be unmatched if it is too small.\n'
         'Annular rectification aligns the seam; defect labels are used only for evaluation.',ha='center',fontsize=10)
fig.subplots_adjust(left=.10,right=.985,top=.88,bottom=.095,wspace=.12,hspace=.27)
fig.savefig(OUT/'03_circular_detection_detail.png',dpi=160);plt.close(fig)

lines=['# Circular weld transfer examples','',
    'Synthetic 80 mm diameter ring, 1921 x 1921 Cartesian samples at 0.05 mm/pixel.',
    'One background and acquisition regime; these examples are not a population benchmark or measured-weld validation.',
    'The 8-10 mm process-keyhole geometry has not been introduced or calibrated by this run.', '',
    'The existing circular profile and pore injection run before image formation. Only the top two z planes used by recessed top-surface rendering are allocated, and detector quadrature is batched. Both matched the native implementation exactly in the saved numerical checks.',
    'New periodic angular cubic fields vary background radial width and relief. Frozen straight campaign files and model weights are unchanged.', '',
    'Both CV methods receive 64x64 annular crops with 0.05 mm radial and nominal centreline arc sampling. Actual tangential spacing changes by radius/40 across the band (about +/-4%). The 78 sectors have 3.222 mm centreline spacing and small ~0.022 mm gaps between their nominal 3.2 mm footprints. Median context uses the four distinct neighbouring sectors and wraps at the circle boundary.',
    'Only the band within about 1.6 mm of the nominal centreline is inspected. Defect labels are used only for evaluation and overlays.', '',
    '| Case | Method | Matched defects | Missed defects | Extra boxes | Negative sectors flagged |',
    '|---|---|---:|---:|---:|---:|']
for level,kind in [(0,'clean'),(5,'clean'),(10,'clean'),(10,'keyhole_pore'),(10,'intrinsic_pore')]:
    meta,*_=get(f'a{level:02d}_{kind}')
    for method in ('cnn','median'):
        m=meta['metrics'][method]
        lines.append(f"| {kind}, {level}% | {method} | {m['tp']} | {m['fn']} | {m['fp']} | {m['clean_false_alarms']}/{m['clean_targets']} |")
lines += ['', '## Reproduce', '', 'Run `WeldDefectDetection/circular_examples_20260928.py`, then `WeldDefectDetection/plot_circular_examples_20260928.py` with the project Python environment.',
          f'Raw arrays, labels, metrics and provenance are in `{DATA}`.',
          f'Acquisition model: {protocol.get("acquisition_model", "legacy")}. This is synthetic evidence, not measured BSE validation.']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
print(OUT)
