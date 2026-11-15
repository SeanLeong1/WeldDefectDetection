"""Oracle diagnostic of reference bias, using paired noise-free validation images."""
from pathlib import Path
import numpy as np
import median_exploration_20261115 as ex

def main():
    ex.verify(True)
    rows=[r for r in ex.read(ex.OLD/'datasets/val/index.json') if r['repeat']==0 and r['count_multiplier']==64 and r['source_class']!='clean']
    records=[]
    for r in rows:
        with np.load(r['geometry_path']) as z:pore=ex.normalized(z['quiet']);mask=z['mask'].astype(bool)
        with np.load(Path(r['geometry_path']).with_name('clean_geometry.npz')) as z:clean=ex.normalized(z['quiet'])
        measured=ex.normalized(np.load(r['raw_path']))
        effect=pore-clean;rms=lambda x:float(np.sqrt(np.mean(x*x)))
        record=dict(scene_id=r['scene_id'],background_id=r['background_id'],diameter_mm=r['diameter_mm'],pore_effect_rms_inside=rms(effect[mask]))
        for kind in ('original','short','gauss'):
            clean_res=ex.residual(clean,kind);pore_res=ex.residual(pore,kind)
            clean_ref=clean-clean_res;pore_ref=pore-pore_res
            bias=rms(clean_res[mask]);contamination=rms((pore_ref-clean_ref)[mask])
            record[kind]=dict(clean_reference_bias_rms_inside=bias,reference_change_due_to_pore_rms_inside=contamination,
                              bias_to_pore_signal=bias/max(record['pore_effect_rms_inside'],1e-12))
            measured_res=ex.residual(measured,kind)
            for label,d in [('quiet',pore_res),('measured',measured_res)]:
                centre=np.median(d,axis=1,keepdims=True)
                scale=np.maximum(.003,1.4826*np.median(abs(d-centre),axis=1,keepdims=True))
                score=abs(d-centre)/scale
                record[kind][label+'_adaptive_scale_inside']=float(np.broadcast_to(scale,mask.shape)[mask].mean())
                record[kind][label+'_score_peak_inside']=float(score[mask].max())
        records.append(record)
    summaries={k:{field:dict(median=float(np.median([r[k][field] for r in records])),q10_q90=np.quantile([r[k][field] for r in records],[.1,.9]).tolist()) for field in records[0][k]} for k in ('original','short','gauss')}
    reduction=[r['short']['clean_reference_bias_rms_inside']/max(r['original']['clean_reference_bias_rms_inside'],1e-12) for r in records]
    scale_ratios={label:float(np.median([r['original'][label+'_adaptive_scale_inside']/r['short'][label+'_adaptive_scale_inside'] for r in records])) for label in ('quiet','measured')}
    above={label:{kind:sum(r[kind][label+'_score_peak_inside']>=16 for r in records) for kind in ('original','short','gauss')} for label in ('quiet','measured')}
    result=dict(scope='64 paired pore geometries from 32 validation backgrounds, with quiet and measured64 repeat0 images. Opening masks locate diagnostic samples only, never inference. Normalisation and sigma=1 preprocessing fixed.',
                summaries=summaries,median_paired_short_to_long_bias_ratio=float(np.median(reduction)),short_lower_bias_geometries=sum(r<1 for r in reduction),geometries=len(records),rows=records,script_sha256=ex.sha(Path(__file__)))
    result.update(median_long_to_short_adaptive_scale_ratio=scale_ratios,geometries_peak_inside_above16=above)
    ex.write(ex.OUT/'reference_bias_diagnostic.json',result)
    print({k:v for k,v in result.items() if k not in ('rows','script_sha256')})

if __name__=='__main__':main()
