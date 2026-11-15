"""Explain one previously identified v4 miss; no model selection."""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import campaign as c
from circular_study import metrics
from inspection.engine import FrozenInspector,Options,load_image


def run():
    torch.set_num_threads(4);scene='test-000-intrinsic_pore';path=c.g.OUT/'datasets/test'/(scene+'.npz')
    inspector=FrozenInspector(c.g.OUT);item=load_image(path,Options(mode='raw',pixel_size_mm=.05),inspector.norm)
    with np.load(path) as z:truth=z['mask'][16:80]
    gt=metrics.regions(truth)[1];assert len(gt)==1
    centre=(gt[0][0]+gt[0][2])//2;cols=np.arange(centre-32,centre+32)%5024
    fig,axes=plt.subplots(2,3,figsize=(10,6),layout='constrained');rows=[]
    for i,method in enumerate(('cnn','median')):
        score=inspector.score(item,method);base=inspector.record[method]['selection']['best']
        axes[i,0].imshow(item['target'][:,cols],cmap='gray',vmin=0,vmax=1)
        axes[i,0].contour(truth[:,cols],levels=[.5],colors=['lime'],linewidths=1)
        axes[i,0].set_title(f'{method}: TOPO and true mask')
        for j,area in enumerate((15,1),1):
            mask,boxes=metrics.binary(score,base['threshold'],area,base['closing'])
            best=max((metrics.circular_iou(b,gt[0],5024) for b in boxes),default=0.)
            row=dict(method=method,min_area=area,threshold=base['threshold'],closing=base['closing'],
                best_box_iou=float(best),correctly_localized=bool(best>=.5),
                threshold_pixels_in_truth=int(((score>=base['threshold'])&(truth>0)).sum()),
                final_pixels_in_truth=int(((mask>0)&(truth>0)).sum()),whole_ring_regions=len(boxes),truth_area=int(truth.sum()))
            rows.append(row);axes[i,j].imshow(mask[:,cols],cmap='gray',vmin=0,vmax=1)
            axes[i,j].contour(truth[:,cols],levels=[.5],colors=['lime'],linewidths=1)
            axes[i,j].set_title(f'Area >= {area}: best box IoU {best:.3f}')
        for ax in axes[i]:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Previously identified intrinsic-pore miss, test-000\nGreen: ground truth; white: predicted mask. Fixed threshold and closing.')
    folder=c.ROOT/'docs/figures/circular_robustness_20261004';folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/'small_pore_failure.png',dpi=160);plt.close(fig)
    c.write(c.OUT/'investigation/fixed_failure_case.json',dict(scene_id=scene,rows=rows,
        scope='Post-hoc illustration of a known v4 error, not a randomly sampled performance estimate or new operating-point choice.'))
    print(rows,flush=True)


if __name__=='__main__':run()
