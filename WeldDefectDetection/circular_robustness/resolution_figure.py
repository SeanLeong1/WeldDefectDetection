"""Show every common-output component-resolution comparison."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import campaign as c


def run():
    rows=c.read(c.OUT/'investigation/resolution.json')['rows']
    folder=c.ROOT/'docs/figures/circular_robustness_20261004';folder.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained',sharey=True)
    for ax,kind in zip(axes,('keyhole_pore','intrinsic_pore')):
        for pitch,color,marker,shift in [(.05,'#c85b28','o',-.006),(.025,'#176a97','^',.006)]:
            values=[r for r in rows if r['kind']==kind and r['pitch_mm']==pitch]
            ax.scatter([r['diameter_mm']+shift for r in values],[100*r['relative_signal_error'] for r in values],
                color=color,marker=marker,label=f'{pitch:g} mm grid',alpha=.75)
        ax.set(title=kind.replace('_',' '),xlabel='Specified pore diameter (mm)',xticks=[.35,.5,.8],ylim=(0,15))
        ax.axvspan(.3,.45,color='gray',alpha=.10);ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Relative component-signal error (%)')
    fig.suptitle('Reference grid 0.0125 mm; common 0.05 mm output and PSF\nEach point is one background/orientation; shaded diameter is outside training range')
    fig.savefig(folder/'resolution.png',dpi=160);plt.close(fig)


if __name__=='__main__':run()
