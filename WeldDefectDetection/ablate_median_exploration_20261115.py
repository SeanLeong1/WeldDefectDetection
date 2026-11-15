"""Validation-only, one-factor removals from the selected classical revision."""
from pathlib import Path
import median_exploration_20261115 as ex

def main():
    ex.verify(True);selected=ex.read(ex.OUT/'frozen.json')['policies']['classical']['op']
    controls={
      'selected':selected,
      'long_references':{**selected,'method':'original_adaptive'},
      'static_calibration':{**selected,'method':'short_static'},
      'no_fragment_merge':{**selected,'merge':False},
      'original_box_extent':{**selected,'shape':'original'},
      'minimum_area_8':{**selected,'area':8},
      'published_baseline':ex.ORIGINAL_OP}
    results={k:[] for k in controls}
    for row in ex.read(ex.OLD/'datasets/val/index.json'):
        cached=ex.read(ex.OUT/'validation_components'/f"{row['scene_id']}.json")
        for name,op in controls.items():
            key=f"{op['method']}:{op['threshold']:g}:{op['closing']}"
            pred=ex.reconstruct(cached['components'][key],op)
            results[name].append(ex.result_row(cached['meta'],cached['truth'],pred))
    payload=dict(created_utc=ex.utc(),scope='Descriptive validation-only one-factor removals after selection. No new model selection and no confirmation use. Effects need not add because factors interact.',controls=controls,
                 results={k:dict(**ex.summary(rows),rows=rows) for k,rows in results.items()},script_sha256=ex.sha(Path(__file__)))
    ex.write(ex.OUT/'validation_ablations.json',payload)
    for k,v in payload['results'].items():print(k,v['macro_f1'],{m:round(b['f1'],3) for m,b in v['by_count'].items()})

if __name__=='__main__':main()
