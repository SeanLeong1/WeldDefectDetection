"""Report detection-threshold changes separately from relaxed scoring."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'WeldDefectDetection'))
import run_median_thresholds_20261019 as study


def name(method,chosen):
    if method=='hard_negative':return 'Hard-negative CNN'
    if method=='coverage':return 'Coverage CNN'
    threshold=int(method.split('_')[1])
    suffix=' (original)' if threshold==12 else ''
    if threshold==chosen:suffix+=' (validation-selected)'
    return f'Median, score >= {threshold}'+suffix


def main():
    frozen=study.verify();audit=study.read(study.OUT/'audit.json');assert audit['status']=='passed'
    regression=study.read(study.OUT/'historical_regression_check.json');assert regression['status']=='passed'
    data=study.read(study.OUT/'results.json');results=data['results'];chosen=frozen['selected_threshold']
    selected=f'median_{chosen}'
    methods=list(dict.fromkeys(['median_12','median_10','median_8',selected,'coverage','hard_negative']))
    figures=ROOT/'docs/figures/median_thresholds_20261019';figures.mkdir(exist_ok=True,parents=True)
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),layout='constrained')
    y=np.arange(len(methods));width=.34
    for ax,pop,title in zip(axes,('primary_pores','all_conditions'),('Clean + two pore classes','Including bulges')):
        for offset,iou,color in [(-width/2,'0.50','#346c9c'),(width/2,'0.25','#db9b42')]:
            vals=[results[k][iou][pop]['f1'] for k in methods]
            ax.barh(y+offset,vals,height=width,color=color,label=f'Box IoU >= {iou}')
            for yy,value in zip(y+offset,vals):ax.text(value+.012,yy,f'{value:.3f}',va='center',fontsize=9)
        ax.set_yticks(y,[name(k,chosen).replace(' (validation-selected)',' *') for k in methods]);ax.invert_yaxis()
        ax.set_xlim(0,1.14);ax.set_xlabel('Box F1');ax.set_title(title);ax.legend(loc='upper right',fontsize=9)
    fig.suptitle('Same fresh images, same evaluation criterion for median and CNN\n24 independent backgrounds; 96 scenes. * Median threshold selected on validation.',fontsize=12)
    fig.savefig(figures/'comparison.png',dpi=160);plt.close(fig)

    historical=study.read(study.original.OUT/'results.json')
    lines=['# Median threshold sensitivity and CNN comparison, 19 October 2026','',
        f'Campaign: `{study.OUT.name}`. The original median and both CNNs remain unchanged. Validation selected median score threshold **{chosen}**; the original threshold is **12**. All matching criteria and score thresholds were declared before generating this confirmation set.',
        '', f'On fresh confirmation, the original median has all-condition F1 **{results["median_12"]["0.50"]["all_conditions"]["f1"]:.3f}** under strict matching and **{results["median_12"]["0.25"]["all_conditions"]["f1"]:.3f}** under relaxed matching. The hard-negative CNN has F1 **{results["hard_negative"]["0.50"]["all_conditions"]["f1"]:.3f}** and **{results["hard_negative"]["0.25"]["all_conditions"]["f1"]:.3f}**, respectively. These pairs use the same images and matching criteria.',
        '', '## What changed','',
        '1. Detection threshold: score cutoffs 3, 4, 6, 8, 10 and 12, keeping the existing eight-neighbour median, radial correction, Gaussian residual smoothing, clean-training calibration, minimum area 15 and closing size 3.',
        '2. Evaluation threshold: strict box IoU >= 0.50 and secondary relaxed box IoU >= 0.25, applied to exactly the same predictions and to every method. Relaxing IoU does not change the detector or its pixel mask.',
        '3. Comparison: both original validation-selected CNNs, with their frozen checkpoints and operating points, evaluated on these same fresh images. No CNN retraining or confirmation-based threshold selection.',
        '', '## Fresh confirmation comparison','',
        '| Method | Pore F1, IoU .50 | Pore F1, IoU .25 | All F1, IoU .50 | All F1, IoU .25 | Clean rings flagged |',
        '|---|---:|---:|---:|---:|---:|']
    for method in methods:
        strict=results[method]['0.50'];loose=results[method]['0.25'];clean=strict['all_conditions']
        lines.append(f'| {name(method,chosen)} | {strict["primary_pores"]["f1"]:.3f} | {loose["primary_pores"]["f1"]:.3f} | {strict["all_conditions"]["f1"]:.3f} | {loose["all_conditions"]["f1"]:.3f} | {clean["clean_rings_flagged"]}/24 |')
    lines+=['','![Comparison](figures/median_thresholds_20261019/comparison.png)',
        '', '## Every prespecified median threshold','',
        'This sensitivity table is descriptive. The primary threshold remains the one chosen on validation, regardless of which threshold looks best below.',
        '', '| Score threshold | Pore F1 .50 | Pore F1 .25 | All F1 .50 | All F1 .25 | Pores matched .50 / 48 | Pores matched .25 / 48 | Clean rings flagged / 24 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for t in study.THRESHOLDS:
        a=results[f'median_{t}']['0.50'];b=results[f'median_{t}']['0.25']
        lines.append(f'| {t}'+(' **selected**' if t==chosen else '')+f' | {a["primary_pores"]["f1"]:.3f} | {b["primary_pores"]["f1"]:.3f} | {a["all_conditions"]["f1"]:.3f} | {b["all_conditions"]["f1"]:.3f} | {a["primary_pores"]["tp"]} | {b["primary_pores"]["tp"]} | {a["all_conditions"]["clean_rings_flagged"]} |')
    lines+=['','## Precision, recall and counts','',
        '| Method | IoU | Population | Precision | Recall | TP | FP | FN |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for method in methods:
        for iou in ('0.50','0.25'):
            for pop in ('primary_pores','all_conditions'):
                r=results[method][iou][pop]
                lines.append(f'| {name(method,chosen)} | {iou} | {pop} | {r["precision"]:.3f} | {r["recall"]:.3f} | {r["tp"]} | {r["fp"]} | {r["fn"]} |')
    lines+=['','## Paired uncertainty','',
        'Differences use 2,000 paired bootstrap replicates, resampling independent backgrounds within dwell strata. These exploratory intervals describe this finite synthetic experiment.',
        '', '| Comparison | IoU | Population | F1 difference | 95% interval |',
        '|---|---:|---|---:|---|']
    for comparison,ious in data['paired_differences'].items():
        for iou,pops in ious.items():
            for pop,values in pops.items():
                v=values['f1'];lo,hi=v['interval_95']
                lines.append(f'| {comparison} | {iou} | {pop} | {v["difference"]:+.3f} | [{lo:+.3f}, {hi:+.3f}] |')
    lines+=['','## Validation selection','',
        'Only the score threshold was reselected. The objective was strict box F1 on clean plus both pore classes, with ties broken by pixel IoU and fewer false regions. Threshold 12 remained eligible, so the procedure did not force a lower setting or deliberately weaken the baseline.',
        '', '| Threshold | Validation pore F1 .50 | Validation all F1 .50 | Clean rings flagged / 24 |',
        '|---|---:|---:|---:|']
    val=study.read(study.OUT/'validation_results.json')
    for t in study.THRESHOLDS:
        r=val['trials'][str(t)]['0.50']
        lines.append(f'| {t} | {r["primary_pores"]["f1"]:.3f} | {r["all_conditions"]["f1"]:.3f} | {r["all_conditions"]["clean_rings_flagged"]} |')
    lines+=['','## Historical CNN results, separate population','',
        'The previously reported acquisition-campaign results used 120 scenes across 24 backgrounds, including duplicate hidden-negative controls. The table below reproduces that historical population and is not combined with the fresh results above.',
        '', '| Method | F1 | Precision | Recall | Clean rings flagged |',
        '|---|---:|---:|---:|---:|']
    for label,key in [('Hard-negative CNN',historical['selected']['hard_negative']),('Coverage CNN',historical['selected']['coverage']),('Median','median')]:
        r=historical['results'][key]['metrics']
        lines.append(f'| {label} | {r["f1"]:.3f} | {r["precision"]:.3f} | {r["recall"]:.3f} | {r["clean_rings_flagged"]}/24 |')
    lines+=['','## Decision and CNN repeatability','',
        'Retain median threshold 12 as the baseline. It won the prospective validation selection, and lower thresholds did not improve fresh F1 under either matching criterion. Keep IoU 0.50 as the primary criterion and report IoU 0.25 separately as localisation sensitivity. The relaxed result is not an algorithm improvement.',
        '', 'Both CNNs outperform the median on this fresh set under both criteria. Their paired F1 difference intervals exclude zero. However, their scores are lower than the historical values, and coverage now outperforms hard-negative mining. The historical 0.923 is a result on one confirmation draw, not a guaranteed repeatable level of performance.',
        '', f'To check the new evaluator after this drop, {regression["method_scene_checks"]} method-scene combinations from four historical clean/pore scenes were rerun. All reproduced the saved predictions, pixel counts and strict detection counts exactly. Checkpoints, renderer sources and normalization also passed hash checks. This verifies consistency on those checked cases; the lower scores belong to the fresh backgrounds.',
        '', 'The CNN advantage is supported, but the relative ranking of the two CNN training strategies is not stable across these confirmation draws. Neither CNN default nor threshold was changed using the new test results.']
    lines+=['','## Interpretation and limits','',
        '- A lower score threshold changes predictions. A lower IoU requirement changes the definition of a successful match. These effects must be reported separately.',
        '- The primary population has 72 scenes: 24 clean and 48 single-pore scenes. The secondary population adds 24 single-bulge scenes, giving 96 scenes and 72 defects. Related scenes remain grouped by background.',
        '- F1 uses object counts, 2 TP / (2 TP + FP + FN). Precision is TP / (TP + FP); recall is TP / (TP + FN). Clean-ring alarms count clean images with at least one predicted region.',
        '- A relaxed box match still does not prove correct segmentation, measured depth or defect classification. Pixel IoU is unchanged when only matching IoU is relaxed.',
        '- Threshold selection is independent of fresh confirmation. All six thresholds are shown; no post-test operating point is substituted.',
        '- Both CNNs were trained on the earlier raised-circular acquisition campaign. These are fresh synthetic backgrounds from the same generator and declared ring coordinates, not physical validation or unknown-image localisation tests.',
        '- The median should be a fair, reproducible comparator. Its settings are not chosen to manufacture a CNN advantage.',
        '', '## Reproduce','', '```powershell',
        '.venv\\Scripts\\python.exe WeldDefectDetection\\run_median_thresholds_20261019.py all',
        '.venv\\Scripts\\python.exe WeldDefectDetection\\verify_median_thresholds_regression_20261019.py',
        '.venv\\Scripts\\python.exe WeldDefectDetection\\report_median_thresholds_20261019.py', '```','',
        f'Artifacts: `{study.OUT}`. Protocol, validation trials, frozen sources/checkpoints, fresh dataset, per-scene predictions, paired intervals and audit are preserved.']
    path=ROOT/'docs/MEDIAN_THRESHOLD_COMPARISON_20261019.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    study.write(study.OUT/'report_artifacts.json',dict(report=str(path),sha256=study.sha(path),
        figures={str(p):study.sha(p) for p in figures.glob('*.png')}))
    print(path,flush=True)


if __name__=='__main__':main()
