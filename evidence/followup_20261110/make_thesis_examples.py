"""Render the two previously selected examples as readable 2-by-2 thesis panels.

Run from the Codebase2 root with .venv/Scripts/python.exe and this script path.
Reads frozen outputs only; writes figures and provenance inside the thesis.
"""
from pathlib import Path
import json, hashlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

THESIS = Path(__file__).resolve().parents[2]
ROOT = THESIS.parents[1]
sys.path.insert(0, str(ROOT/'WeldDefectDetection'))
from circular_study import metrics

OUT = ROOT/'DataSimulation/data/output/median_localisation_20261110_v1'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
sources = [OUT/'datasets/test/index.json']
results = {}
for key in ['original','combined','cnn']:
    p = OUT/'confirmation'/f'{key}.json'
    results[key] = {r['scene_id']:r for r in read(p)['rows']}
    sources.append(p)
meta = {r['scene_id']:r for r in read(sources[0])}
provenance = read(ROOT/'docs/median_localisation_20261110/visual_provenance.json')
plt.rcParams.update({'font.size':14})
figures = []
for index, sid in enumerate(provenance['confirmation_examples'], 1):
    r = meta[sid]; truth = results['combined'][sid]['truth_boxes'][0]
    cx = int((truth[0]+truth[2])/2)%5024
    cy = int((truth[1]+truth[3])/2)
    cols = (np.arange(-24,25)+cx)%5024
    sl = slice(max(0,cy-24),min(128,cy+25))
    raw = np.load(r['raw_path'])[16:144]
    with np.load(r['geometry_path']) as z: mask = z['mask'][16:144]
    sources.extend([Path(r['raw_path']),Path(r['geometry_path'])])
    fig, axes = plt.subplots(2,2,figsize=(9,8.5),layout='constrained')
    for ax, key, title in zip(axes.flat,[None,'original','combined','cnn'],
                             ['TOPO and geometric opening','Original median','Position + size correction','Frozen CNN']):
        ax.imshow(raw[sl][:,cols],cmap='gray')
        if key:
            rec = results[key][sid]
            for x,y,xx,yy in rec['predicted_boxes']:
                width = xx-x
                mid = (((x+xx)/2-cx+2512)%5024-2512)+24
                ax.add_patch(Rectangle((mid-width/2-.5,y-sl.start-.5),width,yy-y,
                                       fill=False,edgecolor='#00cfff',linewidth=2.2))
            best = max((metrics.circular_iou(p,t,5024) for p in rec['predicted_boxes']
                        for t in rec['truth_boxes']),default=0.)
            ax.text(.02,.02,f'Best box IoU: {best:.3f}\nRing TP / FP / FN: {rec["tp"]} / {rec["fp"]} / {rec["fn"]}',
                    transform=ax.transAxes,fontsize=12,va='bottom',
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
        ax.contour(mask[sl][:,cols],[.5],colors=['#ff3366'],linewidths=2)
        ax.set_xlim(-.5,len(cols)-.5)
        ax.set_ylim(raw[sl][:,cols].shape[0]-.5,-.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title,fontsize=14)
    fig.suptitle(f'64× counts; nominal diameter {r["diameter_mm"]:.3f} mm\n'
                 'Red: geometric opening   Cyan: individual detection boxes',fontsize=14)
    target = THESIS/'figures'/f'followup_median_case_{index}.png'
    fig.savefig(target,dpi=180);plt.close(fig)
    figures.append({'scene_id':sid,'figure':str(target.relative_to(THESIS)),
                    'sha256':sha(target)})
record = {'selection':provenance['confirmation_selection'],
          'scope':'Reformatted existing selected examples; no inference or selection changes.',
          'figures':figures,
          'sources':{str(p.relative_to(ROOT)):sha(p) for p in set(sources)},
          'script_sha256':sha(Path(__file__))}
(Path(__file__).parent/'thesis_figure_provenance.json').write_text(json.dumps(record,indent=2))
print('Rendered two 2-by-2 examples from frozen predictions.')
