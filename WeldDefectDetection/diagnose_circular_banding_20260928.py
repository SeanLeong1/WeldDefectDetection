"""Same-render acquisition ablation; no change to frozen data or CV models."""
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import circular_examples_20260928 as c
from bse_renderer.renderer.image_formation import form_channels

OUT=c.ROOT/'docs/figures/circular_banding_diagnostic_20260928'
OUT.mkdir(parents=True,exist_ok=True)
captured={}
original_thermal=c.se.ScanEngine._apply_quadrant_thermal
def thermal(quadrants,**kw):
    captured['before_thermal']=form_channels(quadrants).topo.copy()
    return original_thermal(quadrants,**kw)

class DiagnosticEngine(c.CircularEngine):
    def _apply_adaptive_quadrant_blur(self,quadrants,**kw):
        out=super()._apply_adaptive_quadrant_blur(quadrants,**kw)
        captured['before_acquisition_noise']=form_channels(out).topo.copy()
        return out

cfg=c.load_config_bundle(c.ROOT/'DataSimulation/config')
sample,_=c.sampled(cfg,1360100000,0)
sample=replace(sample,randomize_defect_location=False,slice=replace(sample.slice,slice_offset_mm=0.))
engine=DiagnosticEngine();engine.amplitude=.1
print('Rendering one exact acquisition ablation.',flush=True)
with patch.object(c.se,'build_voxel_volume',c.top_volume),patch.object(c.se,'compute_quadrant_signals',c.chunked_quadrants),patch.object(c.se.ScanEngine,'_apply_quadrant_thermal',staticmethod(thermal)):
    result=engine.run(sample)
gain=float(np.interp(sample.beam.focusing_current_norm,[0,1],[1.10,.92]))
assert engine.calibration.topo.scale==1 and engine.calibration.topo.bias==0
for key in captured:captured[key]*=gain
captured['complete_current_model']=result.channels.topo
with np.load(c.OUT/'a10_clean.npz') as a:
    replay_error=float(np.max(np.abs(a['raw']-result.channels.topo)))
assert replay_error==0,replay_error
y,x=np.indices(result.channels.topo.shape)
radius=np.hypot((x-960)*.05,(y-960)*.05)
background=(radius<35)|(radius>44)
metrics={}
for name,arr in captured.items():
    values=arr[background]
    metrics[name]={'background_mean':float(values.mean()),'background_std':float(values.std()),
                   'background_p01_p99':[float(v) for v in np.quantile(values,[.01,.99])]}
metrics['exact_replay_max_error']=replay_error
metrics['scope']='One synthetic acquisition ablation. Not a calibrated physical BSE validation.'
(OUT/'measurements.json').write_text(json.dumps(metrics,indent=2)+'\n')
norm=json.loads((c.ROOT/'DataSimulation/data/output/variability_campaign_20260926_v1/normalization.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(13.5,5.1))
titles=['Before acquisition noise','Contamination + shot noise','Plus current drift/readout model']
for ax,(name,arr),title in zip(axes,captured.items(),titles):
    image=np.sqrt(np.clip((arr-norm['raw_low'])/(norm['raw_high']-norm['raw_low']),0,1))
    ax.imshow(image,cmap='gray',vmin=0,vmax=1,origin='lower',interpolation='nearest')
    ax.set_title(title,fontsize=12)
    ax.set_xlabel(f"Background SD: {metrics[name]['background_std']:.5f} raw units",fontsize=10)
    ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('Where the large bands enter the circular TOPO image',fontsize=17,y=.98)
fig.text(.5,.075,'Same 80 mm specimen, detector response and intensity mapping in every panel.\n'
         'Left retains fixed detector sensitivity imbalance. Right reproduces the previously shown image exactly.',ha='center',fontsize=10)
fig.subplots_adjust(top=.87,bottom=.22,left=.02,right=.98,wspace=.08)
fig.savefig(OUT/'banding_ablation.png',dpi=170);plt.close(fig)
np.savez_compressed(OUT/'staged_topo.npz',**captured)
print(json.dumps(metrics,indent=2),flush=True)
