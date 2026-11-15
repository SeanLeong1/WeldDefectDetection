"""Fixed first-background examples; never select examples by detector success."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from . import campaign as c


def run():
    folder=c.ROOT/'docs/figures/circular_robustness_20261004';folder.mkdir(parents=True,exist_ok=True)
    norm=c.read(c.g.OUT/'normalization.json');pitch=2*np.pi*40/5024
    cols=np.arange(-320,320);fig,axes=plt.subplots(7,2,figsize=(14,11),layout='constrained')
    for i,variant in enumerate(c.VARIANTS):
        sid=f'robust-000-{variant}'
        with np.load(c.OUT/'datasets'/(sid+'.npz')) as z:
            raw=z['raw'][0,32:96][:,cols%5024]
            instances=z['instances'][:,32:96][:,:,cols%5024]
        x=np.sqrt(np.clip((raw-norm['low'])/(norm['high']-norm['low']),0,1))
        truth=np.zeros((*x.shape,3))
        for instance,color in zip(instances,[(0,.8,.3),(.8,.1,.8),(.1,.5,1.)]):
            truth+=instance[:,:,None]*np.asarray(color)
        pred=c.read(c.OUT/'predictions_native'/(sid+'.json'))
        for ax,img,method,color in zip(axes[i],(x,np.clip(truth,0,1)),('cnn','median'),('red','orange')):
            ax.imshow(img,cmap='gray',vmin=0,vmax=1,origin='lower',extent=[-320.5*pitch,319.5*pitch,-1.6,1.6],aspect='auto')
            for a,b,cc,d in pred[f'nominal/{method}/best']['predicted_boxes']:
                if a>2512:a-=5024;cc-=5024
                if cc>=-320 and a<=320:
                    ax.add_patch(Rectangle(((a-.5)*pitch,(b-32)*.05),(cc-a)*pitch,(d-b)*.05,fill=False,edgecolor=color,linewidth=1.2))
            ax.set_xlim(-320*pitch,320*pitch);ax.set_ylabel('Radial offset (mm)')
            ax.set_title(f'{variant}: '+('TOPO + CNN boxes' if method=='cnn' else 'Individual labels + median boxes'),fontsize=10)
    for ax in axes[-1]:ax.set_xlabel('Arc length from the periodic join (mm)')
    fig.suptitle('Fixed background 000, all seven layouts; label colors blend in overlap',fontsize=13)
    fig.savefig(folder/'paired_layouts.png',dpi=160);plt.close(fig)


if __name__=='__main__':run()
