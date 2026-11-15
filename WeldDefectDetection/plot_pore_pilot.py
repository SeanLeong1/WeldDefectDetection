"""Readable scientific pilot overview from stored counterfactual arrays."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
if not (ROOT/'pilot_pairs').exists():
 ROOT=ROOT.parent/'DataSimulation/data/output/pore_campaign_20260921_v1'
rows=[r for r in json.loads((ROOT/'pilot_pairs/report.json').read_text())['rows'] if int(r['sequence_id'].split('-')[-1]) in (9,13,17,18,22,26)]
records={r['sample_id']:r for r in [json.loads(s) for s in (ROOT/'datasets/pilot/samples.jsonl').read_text().splitlines()]}
fig,axes=plt.subplots(len(rows),5,figsize=(12,14))
for ax,row in zip(axes,rows):
 r=records[row['sample_id']]
 with np.load(ROOT/'pilot_pairs'/(r['sequence_id']+'.npz')) as a:
  i=list(a['sample_ids']).index(row['sample_id']);on,off,mask=a['raw'][i],a['off'][i],a['mask'][i]
  noisy=np.load(ROOT/'datasets/pilot'/r['raw_channels']['topo'])
  low=float(min(noisy.min(),on.min(),off.min()));high=float(max(noisy.max(),on.max(),off.max()))
  for j,img in enumerate((off,on,noisy,abs(on-off),mask)):
   ax[j].imshow(img,cmap='gray' if j<3 else 'magma',**(dict(vmin=low,vmax=high) if j<3 else {}));ax[j].set_xticks([]);ax[j].set_yticks([])
  p=row['parameters'];ax[0].set_ylabel(f"{row['kind']}\ndiameter {p['diameter_mm']:.3f} mm\ndepth parameter {p['depth_mm']:.3f} mm",fontsize=8)
for ax,title in zip(axes[0],('No defect','Noiseless pore','Noisy pore','Absolute change*','Geometric mask')):ax.set_title(title,fontsize=10)
fig.suptitle('Pore pilot: representative grid points; full grids preserved separately',y=.99)
fig.text(.5,.015,'*Difference panels individually scaled. Raw-image columns share a scale within each row. No cases were removed.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.035,1,.96));fig.savefig(ROOT/'pilot_overview.png',dpi=140);plt.close(fig)
