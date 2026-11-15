"""Create fixed-order error examples without changing any model settings."""
import sys
from pathlib import Path
REPO=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(REPO/'WeldDefectDetection'),str(REPO/'DataSimulation')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from box_refinement.campaign import DEFAULT,ARMS,read,check,require
from box_comparison.core import load_windows,write

if __name__=='__main__':
    root=DEFAULT;check(root);require(root,'evaluate')
    data=load_windows(root/'confirmation');indices={m['window_id']:i for i,m in enumerate(data['meta'])}
    fig,axes=plt.subplots(4,2,figsize=(7,12));records=[]
    for ri,arm in enumerate(ARMS):
        rows=read(root/'predictions'/f'{arm}.json')
        for col,error in enumerate(('fp','fn')):
            row=next((r for r in rows if r[error]>0),None);ax=axes[ri,col]
            if row is None:
                ax.text(.1,.5,f'No {error} case');ax.axis('off');continue
            ax.imshow(data['target'][indices[row['window_id']]],cmap='gray',vmin=0,vmax=1,extent=(0,64,64,0))
            for key,color in [('truth_boxes','lime'),('boxes_xyxy','orange')]:
                for x0,y0,x1,y1 in row[key]:ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=color,lw=1))
            ax.set_title(f'{arm}: first {error} case\n{row["window_id"]}',fontsize=8);ax.axis('off')
            records.append(dict(approach=arm,error=error,window_id=row['window_id'],source_class=row['source_class'],
                true_regions=row['truth_boxes'],predicted_boxes=row['boxes_xyxy'],tp=row['tp'],fp=row['fp'],fn=row['fn']))
    fig.tight_layout();fig.savefig(root/'error_examples.png',dpi=160);plt.close(fig)
    write(root/'error_examples.json',records)
    (root/'ERROR_EXAMPLES.md').write_text('# Remaining localisation errors\n\nThe first false-positive and missed-region case in saved target order is shown for each approach. A case can have both types of error. Green denotes ground truth and orange predictions. No model settings were changed to produce this panel.\n\n![Errors](error_examples.png)\n',encoding='utf-8')
