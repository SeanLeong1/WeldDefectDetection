"""Publish a current teaching edition without altering the historical hashed guide."""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(REPO/'WeldDefectDetection'),str(REPO/'DataSimulation')]
from box_comparison.campaign import read
from box_comparison.core import sha,write
from box_development.campaign import DEFAULT,OLD,ARMS,check,require


def build(root=DEFAULT):
    check(root);require(root,'verify')
    source=REPO/'DataSimulation/PROJECT_PIPELINE_FROM_CASINO_TO_DEFECT_DETECTION.md'
    bound=read(OLD/'FINAL_CAMPAIGN_MANIFEST.json')['teaching_guide']
    if sha(source)!=bound['sha256']:raise ValueError('Historical teaching guide changed.')
    result=read(root/'results.json');frozen=read(root/'frozen_models.json')
    populations=read(root/'training_populations.json');curves=read(root/'learning_curve.json')
    relative='data/output/'+root.name
    intro=['# From CASINO to TOPO defect boxes: current project guide','',
        '**Current CV edition: 21 September 2026.** The active comparison has four approaches: thresholding, target-only CNN, neighbour-median residual and CNN + median. They are evaluated on the same 64x64 target crops and produce bounding boxes. The earlier learned context predictor remains historical.',
        f'The five-step development campaign is complete. Its [artifact-generated report]({relative}/REPORT.md), [error and visibility diagnostics]({relative}/DIAGNOSTICS.md) and [development teaching chapter](../WeldDefectDetection/FIVE_STEP_DEVELOPMENT.md) contain the current evidence. Section 28 below records its dataset sizes, method choices and fresh confirmation results.',
        'The renderer still uses assumed constant B = 0.9206948532270666 for the synthetic 304SS, 60 keV workflow. No new physical emission qualification or measured-SEM validation was established.',
        'Sections 1-27 retain the chronological CASINO, geometry, rendering and earlier evaluation explanation. Their dated numerical tables belong to those earlier experiments. The current box comparison uses a different, explicitly matched target population, so its scores must not be substituted into the earlier static/spatial benchmark tables.',
        'The original `PROJECT_PIPELINE_FROM_CASINO_TO_DEFECT_DETECTION.md` remains unchanged because the historical campaign manifest binds its hash. This current edition is a separate document.','']
    old=source.read_text(encoding='utf-8')
    old=old.replace('# From CASINO Electron Trajectories to Synthetic Weld-Defect Detection','## Foundation and historical evidence',1)
    old=old.replace('**Active workflow, 15 September 2026:**','**Historical workflow, 15 September 2026:**',1)
    old=old.replace('**Teaching edition, 15 September 2026.**','**Historical teaching edition, 15 September 2026.**',1)
    chapter=['','## 28. The current four-method bounding-box comparison','',
        '### 28.1 Audit the image-to-label relationship','',
        'The renderer supplies a geometric mask. Each eight-connected component receives a half-open bounding box. A successful detection requires one-to-one matching at IoU at least 0.5. No small or weak truth regions are removed.',
        'The new audit replayed 18 development sequences, covering 19 positive target crops, with defects present and removed. All masks were reproduced. Several sampled classes had little mask area whose noiseless change exceeded a descriptive two-noise-standard-deviation threshold. This small audit measures a renderer response and does not prove that an entire defect class is undetectable.',
        '### 28.2 Test simple residuals and neighbouring context','',
        'Thresholding was tested with several local background scales. Median context was tested with smooth neighbour-profile correction and a frozen clean-training variation map. Validation retained the old configuration when a new feature did not improve box F1. References always exclude the target and use four non-overlapping adjacent crops.',
        '### 28.3 Add independent training scenes','',
        '| Training size | Sequences | Targets | Defect-containing targets |','|---|---:|---:|---:|']
    for key,p in populations.items():chapter.append(f'| {key}x | {p["sequences"]} | {p["targets"]} | {p["positive_targets"]} |')
    chapter+=['','All added data were designated training before generation. Original validation and test windows remained held out. Three seeds per family were trained at every size for 20 epochs. The normalization remained frozen from the original training set.',
        '', '| Method | Mean validation F1, 1x | 2x | 4x |','|---|---:|---:|---:|']
    for arm in ('cnn','cnn_median'):
        means=[]
        for n in (1,2,4):
            values=[r['metrics']['box_f1'] for r in curves if r['arm']==arm and r['multiplier']==n]
            means.append(sum(values)/len(values))
        chapter.append('| '+arm+' | '+' | '.join(f'{v:.3f}' for v in means)+' |')
    chapter+=['','These are development results, not independent performance estimates. Fixed epochs mean that larger datasets receive more optimizer steps. The experiment measures the practical recipe, not a pure data-only effect at matched compute.',
        '### 28.4 Test box supervision','',
        'Both CNN families met the prespecified extent-error condition. A second output was trained on rectangles derived from the unchanged masks, with auxiliary loss weight 0.25. Validation chose the output used for boxes; pixel IoU continued to use the segmentation output. This did not introduce a new ground-truth definition or relax the success threshold.',
        '### 28.5 Freeze and independently confirm','',
        f'The primary method was frozen as **{frozen["primary"]}** before generating 70 new sequences and 350 confirmation targets. Actual seeds and raw hashes were checked against all development and previous confirmation data.',
        '', '| Method | First campaign, same fresh cases | Second campaign, same fresh cases | Current selected box F1 | Precision | Recall |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        m=result['metrics'][arm]
        chapter.append(f'| {arm} | {result["metrics"]["original_"+arm]["box_f1"]:.3f} | {result["metrics"]["previous_"+arm]["box_f1"]:.3f} | {m["box_f1"]:.3f} | {m["box_precision"]:.3f} | {m["box_recall"]:.3f} |')
    chapter+=['','The paired 95% intervals, class results, shuffled-context controls and simple baselines are in the generated report. An observed increase whose interval includes zero is not established improvement. All comparisons here use the same target population.',
        f'The primary synthetic image/pixel skill gate passed: **{result["synthetic_pixel_image_skill_gate"]}**. That gate is separate from satisfactory bounding-box recall and does not establish real-SEM performance.',
        '### 28.6 Use the frozen methods','',
        'Run or resume the campaign from the repository root:','','```powershell',
        "& '.\\.venv\\Scripts\\python.exe' '.\\WeldDefectDetection\\develop_boxes.py' --stage all",'```','',
        'Use `WeldDefectDetection/infer_developed_boxes.py` for label-free inference. It loads the frozen normalization, checkpoint and component rules, accepts a raw TOPO bundle and exports boxes without probability labels. The campaign includes a verified example for all four approaches.',
        '',f'Current numerical sources: [{relative}/results.json]({relative}/results.json), [{relative}/learning_curve.json]({relative}/learning_curve.json), and [{relative}/training_populations.json]({relative}/training_populations.json).','']
    output=REPO/'DataSimulation/PROJECT_PIPELINE_CURRENT.md'
    output.write_text('\n'.join(intro)+old+'\n'+'\n'.join(chapter),encoding='utf-8')
    snapshot=root/'PROJECT_PIPELINE_CURRENT_SNAPSHOT.md'
    snapshot.write_bytes(output.read_bytes())
    write(root/'current_guide_provenance.json',dict(path=str(output),sha256=sha(output),historical_source_sha256=sha(source),
        snapshot_path=str(snapshot),snapshot_sha256=sha(snapshot),relative_link_base=str(output.parent),
        note='The dated snapshot preserves this edition if the current entry guide is updated in a later campaign.',
        results_sha256=sha(root/'results.json'),learning_curve_sha256=sha(root/'learning_curve.json')))
    print(output)


if __name__=='__main__':build()
