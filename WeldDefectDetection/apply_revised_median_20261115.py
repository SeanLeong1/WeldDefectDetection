"""Apply the frozen follow-up detector to an existing registered raw TOPO array.

This entry point does not load geometry masks or use the true pore position.
Input must use the campaign's physical sampling, coordinates and intensity scale.
"""
from pathlib import Path
import argparse
import numpy as np
import median_exploration_20261115 as ex


def predict(raw, count=None):
    frozen=ex.verify(True)
    raw=np.asarray(raw)
    if raw.shape!=(160,5024) or not np.isfinite(raw).all():
        raise ValueError('Expected a finite 160 x 5024 raw TOPO array in the original registered coordinates.')
    if count is None:
        policy='classical';op=frozen['policies'][policy]['op']
    else:
        if count not in (1,4,64):raise ValueError('Count-specific policy supports count levels 1, 4 or 64 only.')
        policy='per_count';op=frozen['policies'][policy][str(count)]['op']
    score=ex.score_maps(raw,ex.read(ex.OUT/'calibration.json'),{op['method']})[op['method']]
    boxes=ex.reconstruct(ex.components(score,op['threshold'],op['closing']),op)
    full_boxes=[[x,y+16,r,b+16] for x,y,r,b in boxes]
    return dict(policy=policy,count_multiplier=count,settings=op,
                scored_coordinate_boxes=boxes,raw_coordinate_boxes=full_boxes,
                box_format='[left, top, right, bottom], right/bottom exclusive; angular coordinates wrap at 5024 and right may exceed 5024',
                scope='Synthetic campaign sampling and fixed training intensity normalisation; no measured-image calibration.'),score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path,help='Raw 160 x 5024 .npy TOPO file, before normalisation.')
    p.add_argument('--output',required=True,type=Path,help='Prediction JSON output.')
    p.add_argument('--count',type=int,choices=(1,4,64),help='Optional known count level selects the frozen count-specific policy.')
    p.add_argument('--score-output',type=Path,help='Optional 128 x 5024 score .npy output.')
    a=p.parse_args()
    for target in (a.output,a.score_output):
        if target is not None and target.exists():raise FileExistsError(f'Refusing to overwrite {target}')
    result,score=predict(np.load(a.input,allow_pickle=False),a.count)
    ex.write(a.output,result)
    if a.score_output is not None:
        a.score_output.parent.mkdir(parents=True,exist_ok=True);np.save(a.score_output,score)
    print(a.output)


if __name__=='__main__':main()
