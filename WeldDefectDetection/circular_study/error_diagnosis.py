"""Observational error taxonomy after model freeze, never a tuning input."""
import numpy as np,torch
from . import generation as g,metrics,median,training,evaluation


def run():
    torch.set_num_threads(4);frozen=evaluation.verify_freeze();data=g.load('test')
    model=training.load_model(frozen['cnn'])
    score_sets=dict(cnn=training.predict(data,model,frozen['cnn']['config']),median=median.scores(data,frozen['median']['spec']))
    output={};examples={}
    for method,scores in score_sets.items():
        spec=frozen[method]['selection']['best'];rows=[]
        for score,truth,meta in zip(scores,data['mask'],data['meta']):
            binary,boxes=metrics.binary(score,spec['threshold'],spec['min_area'],spec['closing'])
            _,gt=metrics.regions(truth)
            matched=metrics.match(boxes,gt,score.shape[1])
            if not gt:
                category='clean_false_alarm' if boxes else 'clean_correct'
                peak=None;above=0;overlap=0;best_iou=None
            else:
                peak=float(score[truth>0].max());above=int(((score>=spec['threshold'])&(truth>0)).sum())
                overlap=int(((binary>0)&(truth>0)).sum())
                best_iou=max((metrics.circular_iou(p,t,score.shape[1]) for p in boxes for t in gt),default=0.)
                if matched['tp']==len(gt):category='localized'
                elif not above:category='below_threshold_inside_label'
                elif not overlap:category='label_response_removed_by_postprocessing'
                else:category='region_localization_mismatch'
            row=dict(scene_id=meta['scene_id'],source_class=meta['source_class'],category=category,
                area_px=meta['area_px'],peak_score_in_label=peak,threshold=spec['threshold'],
                threshold_pixels_in_label=above,final_pixels_in_label=overlap,best_box_iou=best_iou,
                signal_to_noise_proxy=meta['signal_change_in_mask']/max(meta['noise_rms'],1e-12),**matched)
            rows.append(row)
            if category not in ('localized','clean_correct') and method+'_'+category not in examples:
                box=gt[0] if gt else boxes[0];centre=(box[0]+box[2])//2
                examples[method+'_'+category]=dict(scene_id=meta['scene_id'],centre=int(centre),
                    target=median.tile(data['x'][len(rows)-1,0,16:80],centre).tolist(),
                    mask=median.tile(truth,centre).tolist(),score=median.tile(score,centre).tolist())
        output[method]=dict(rows=rows,counts={k:sum(r['category']==k for r in rows) for k in sorted({r['category'] for r in rows})})
    g.write(g.OUT/'diagnostics/error_taxonomy.json',dict(methods=output,examples=examples,
        scope='Descriptive categories on frozen predictions. Labels are geometric. A low score inside a label does not prove the physical signal is absent; a poor box IoU does not prove the pore was wholly invisible. First example per category is retained. No data exclusion or model update follows.'))
    print({m:r['counts'] for m,r in output.items()},flush=True)


if __name__=='__main__':run()
