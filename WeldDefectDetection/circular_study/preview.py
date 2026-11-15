"""Inspect fixed first-background images before fitting any detector."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import generation as g
from .median import tile


def run():
    folder=g.OUT/'datasets/train';arrays=[]
    for kind in g.baseline.c.KINDS:
        with np.load(folder/f'train-000-{kind}.npz') as a:arrays.append(dict(raw=a['raw'][0,16:80],mask=a['mask'][16:80],quiet=a['quiet'][16:80]))
    lo,hi=np.quantile(arrays[0]['raw'],[.01,.995])
    centre=int(np.median(np.flatnonzero(arrays[1]['mask'].any(0))))
    fig,axes=plt.subplots(3,4,figsize=(13,8),layout='constrained')
    for i,(kind,a) in enumerate(zip(g.baseline.c.KINDS,arrays)):
        norm=np.sqrt(np.clip((a['raw']-lo)/(hi-lo),0,1))
        panels=[norm,tile(norm,centre),tile(a['mask'],centre),tile(a['quiet']-arrays[0]['quiet'],centre)]
        for j,(p,title) in enumerate(zip(panels,['Whole ring','Local TOPO','Geometric mask','Quiet pore minus clean'])):
            axes[i,j].imshow(p,cmap='gray' if j<3 else 'coolwarm',aspect='auto' if j==0 else 'equal')
            axes[i,j].set_title(kind+'\n'+title);axes[i,j].axis('off')
    fig.suptitle('Pre-training fixed first-background inspection; display-only normalization from its clean image')
    path=g.OUT/'diagnostics/preview.png';fig.savefig(path,dpi=140);plt.close(fig);print(path)


if __name__=='__main__':run()
