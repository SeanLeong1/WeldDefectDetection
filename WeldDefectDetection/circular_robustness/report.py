"""Report all predeclared stress conditions and explanatory diagnostics."""
from pathlib import Path
import json,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import campaign as c

def run():
    from .comparisons import run as compare
    from .figures import run as examples
    from .resolution_figure import run as resolution_plot
    from .failure_case import run as failure
    compare()
    examples()
    resolution_plot()
    failure()
    data=c.read(c.OUT/'results.json')['results'];post=c.read(c.OUT/'investigation/postprocessing.json')
    comparisons=c.read(c.OUT/'paired_localisation.json')['results']
    resolution=c.read(c.OUT/'investigation/resolution.json');figdir=c.ROOT/'docs/figures/circular_robustness_20261004';figdir.mkdir(parents=True,exist_ok=True)
    doc=c.ROOT/'docs/CIRCULAR_ROBUSTNESS_RESULTS_20261004.md'
    lines=['# Circular robustness and small-pore investigation','',
        'The existing v4 CNN and neighbouring-median models remain frozen. This extension evaluates 24 fresh independent backgrounds, with seven paired scene layouts each (168 scenes). It does not train a new detector or select a new operating point. The circular recessed profile is retained; unfamiliar weld profiles are paused.',
        '', '## Main findings','',
        '- On the same 24 clean and 48 single-pore scenes, CNN nominal box F1 is 0.938 (45 TP, 3 FP, 3 FN). Across the twenty declared localization conditions, F1 ranges from 0.804 to 0.938. The worst observed condition is a -0.5 mm radius error, with 3/24 clean rings flagged versus 0/24 nominally.',
        '- The paired F1 change for that radius error is -0.134, with a pointwise 95% background-bootstrap interval of -0.260 to -0.014. This is a descriptive comparison among the full reported sweep, not a simultaneous tolerance guarantee.',
        '- Nominal CNN F1 is 0.917 for far pairs, 0.907 for close pairs and 0.931 for triples. Overlapping pairs score 1.000 against merged union regions, while individual-pore recall is only 24/48 (0.500). The perfect union score does not establish separation of the two pores.',
        '- Median nominal F1 on the clean/single population is 0.054, with 5/24 clean rings flagged. A +0.5 mm centre error on either axis flags 24/24 clean rings. The current median method is not robust to those errors in this synthetic population.',
        '- Lowering the area filter alone recovers no additional correctly localized pores on the explanatory v4 test analysis. Finer internal spacing reduces the component-signal discrepancy, but full acquisition-chain convergence remains to be tested.',
        '', 'Zero observed alarms are not evidence of zero risk: the exact two-sided 95% interval for 0/24 clean rings is approximately 0% to 14.2%. Clean-alarm intervals below use exact binomial bounds because ordinary bootstrap intervals collapse to zero when no alarms are observed. The raw bootstrap field is retained in results.json for traceability and must not be treated as a zero-risk bound.',
        '', '## Design and interpretation','',
        '- Each background has clean, single keyhole, single intrinsic, far pair, near pair, overlapping pair and triple scenes.',
        '- Each pair contains one keyhole and one intrinsic pore; the triple contains two keyhole pores and one intrinsic pore. This does not exhaust all pore-family, depth or count combinations.',
        '- Pair centreline separations are 12, 1.4 and 0.35 mm. Triple centres are at -12, 0 and +12 mm. Diameter and fixed radial-offset strata vary across backgrounds; each body keeps its identity and shape across paired layouts.',
        '- Diameter stratum, radial offset and the existing profile-variation amplitude co-vary across backgrounds in this extension. Paired comparisons hold them fixed within a background, but separate causal effects of these three variables cannot be estimated. Zero-angle seam examples also belong to the zero-amplitude stratum. The population is a declared stress design, not a prevalence sample or a fully crossed factorial design.',
        '- Individual masks are saved, including overlap. Primary box matching uses connected union regions. Secondary instance matching requires separate one-to-one boxes for the injected pores. Neither method was trained for instance segmentation; a merged region can pass union matching while failing individual-pore matching.',
        '- Native response composition takes minimum yield, maximum void fraction and adds relief. These are existing surrogate rules, not a validated physical interaction model.',
        '- Twenty ring-estimate conditions cover centre x/y and radius errors of +/-0.1, +/-0.25 and +/-0.5 mm, plus a combined (+0.25,-0.25,+0.25) mm error. All conditions use paired clean/single scenes. Nominal and combined error also evaluate every multi-pore layout.',
        '- Images are resampled into estimated ring coordinates, inferred, and scores mapped back to the fixed true inspection band. Missing coverage scores zero; ground truth is never removed to make an error look easier. This is sensitivity to a specified localization error, not validation of an automatic ring finder.',
        '- Each estimated ring is extracted directly from the native Cartesian TOPO image with a checked context halo. Scores are then mapped to the fixed true evaluation coordinates.',
        '', '### Pre-scoring numerical correction','',
        'A check on the first background and its clean/two single-pore variants found approximately 7% relative input error for centre perturbations when an already unwrapped image was interpolated again. Nominal and the checked radial shift reproduced exactly. Before any stress scores were produced, protocol_native_v2.json replaced that shortcut with direct native Cartesian extraction. The original protocol/source and diagnostic remain preserved. Every one of the 168 native rerenders must reproduce the original raw, quiet, union and individual-mask arrays exactly. No model, threshold, scene, perturbation or population was changed by this amendment.',
        '', '## Nominal whole-ring results','',
        '| Scene layout | Method | Union TP/FP/FN | Union F1 | Individual pores located | Individual F1 |','|---|---|---:|---:|---:|---:|']
    for variant in c.VARIANTS:
        for method in ('cnn','median'):
            r=data[f'nominal/{method}/best']['per_variant'][variant]
            lines.append(f'| {variant} | {method} | {r["tp"]}/{r["fp"]}/{r["fn"]} | {r["f1"]:.3f} | {r["instance_tp"]}/{r["instance_tp"]+r["instance_fn"]} | {r["instance_f1"]:.3f} |')
    lines+=['','The clean row has no positive labels, so its F1 is conventionally zero. Use its false regions and the following ring-alarm counts instead. Results for the separately frozen recall-focused setting are retained in results.json. A better secondary test score does not change the default operating point.',
        '', '### Multiple pores with combined ring-estimate error','',
        '| Layout | Method | Nominal union F1 | Combined-error union F1 | Nominal individual recall | Combined-error individual recall |',
        '|---|---|---:|---:|---:|---:|']
    for variant in c.VARIANTS[3:]:
        for method in ('cnn','median'):
            nominal=data[f'nominal/{method}/best']['per_variant'][variant]
            combined=data[f'combined/{method}/best']['per_variant'][variant]
            lines.append(f'| {variant} | {method} | {nominal["f1"]:.3f} | {combined["f1"]:.3f} | {nominal["instance_recall"]:.3f} | {combined["instance_recall"]:.3f} |')
    lines+=[
        '', '![Fixed paired layout examples](figures/circular_robustness_20261004/paired_layouts.png)',
        '', 'The first background is fixed for illustration, with all seven layouts shown around the zero-angle join. This is a local unwrapped view of the circular weld. Whole-ring metrics include every angular column, including false alarms outside this view. Colors identify injected masks, not predictions of pore type.',
        '', '## Imperfect ring localisation','',
        'These rows use the same 24 clean and 48 single-pore scenes in every condition. Bootstrap intervals resample whole backgrounds; the numerous transformed views are not independent welds.',
        '', '| Ring-estimate error | Method | F1 | Recall | Clean rings flagged (exact 95% rate interval) | F1 difference from nominal (paired 95% interval) | Label pixels outside inspection band |','|---|---|---:|---:|---:|---:|---:|']
    for e in c.PERTURBATIONS:
        for method in ('cnn','median'):
            r=data[f'{e["name"]}/{method}/best']['single_pore_population']
            diff=comparisons[f'{e["name"]}/{method}'];lo,hi=diff['paired_percentile_95']
            alo,ahi=diff['clean_alarm_exact_95']
            lines.append(f'| {e["name"]} | {method} | {r["f1"]:.3f} | {r["recall"]:.3f} | {r["clean_rings_flagged"]}/24 ({alo:.1%}, {ahi:.1%}) | {diff["f1_difference"]:+.3f} ({lo:+.3f}, {hi:+.3f}) | {r["truth_pixels_outside_inspection_band"]} |')
    lines+=['','Intervals are pointwise, not simultaneous bounds across all twenty conditions. Small sample uncertainty and repeated views limit any general robustness claim.']
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,axis in zip(axes,('dx','dy','dr')):
        for method in ('cnn','median'):
            x=[-.5,-.25,-.1,0,.1,.25,.5]
            vals=[data[f'{"nominal" if v==0 else f"{axis}_{v:+.2f}"}/{method}/best']['single_pore_population']['f1'] for v in x]
            line,=ax.plot(x,vals,'o-',label=method)
            intervals=[data[f'{"nominal" if v==0 else f"{axis}_{v:+.2f}"}/{method}/best']['single_pore_intervals']['f1'] for v in x]
            ax.fill_between(x,[v[0] for v in intervals],[v[1] for v in intervals],color=line.get_color(),alpha=.12)
        ax.set(xlabel=f'{axis} error (mm)',ylabel='Union box F1',ylim=(0,1));ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Frozen models, same synthetic scenes, imperfect ring estimate\nShading: pointwise 95% background-bootstrap intervals')
    fig.savefig(figdir/'localisation.png',dpi=160);plt.close(fig)
    lines+=['','![Localization sensitivity](figures/circular_robustness_20261004/localisation.png)',
        '', '## Small-pore rendering sensitivity','',
        'Two physical backgrounds, two orientations, two pore families and three diameters are evaluated at three internal spacings. Output integration and a 0.05 mm Gaussian PSF are common. Reference is 0.0125 mm. These are component diagnostics with fixed surface planes and unit gains, not the full noisy acquisition chain or proof of absolute convergence. The 0.35 mm diameter is below the training population range.',
        '', '| Diameter | Pore family | 0.05 mm relative signal-error range | 0.025 mm range |','|---|---|---:|---:|']
    for diameter in (.35,.5,.8):
        for kind in ('keyhole_pore','intrinsic_pore'):
            rr=[r for r in resolution['rows'] if r['diameter_mm']==diameter and r['kind']==kind]
            a=[r['relative_signal_error'] for r in rr if r['pitch_mm']==.05];b=[r['relative_signal_error'] for r in rr if r['pitch_mm']==.025]
            lines.append(f'| {diameter} mm | {kind} | {min(a):.2%} to {max(a):.2%} | {min(b):.2%} to {max(b):.2%} |')
    lines+=['','![Component resolution sensitivity](figures/circular_robustness_20261004/resolution.png)',
        '', '## Thresholding, area filtering and localization diagnosis','',
        'This section is explicitly post-hoc analysis of the previously opened v4 test set. It explains mechanisms; it is not new confirmation and does not select a replacement model. The probability/score thresholds stay frozen. Every declared area/closing combination is shown.',
        '', '| Method | Minimum area | Closing | F1 | Recall | Clean rings flagged |','|---|---:|---:|---:|---:|---:|']
    for method,result in post['methods'].items():
        for row in result['settings']:
            m=row['metrics'];lines.append(f'| {method} | {row["min_area"]} | {row["closing"]} | {m["f1"]:.3f} | {m["recall"]:.3f} | {m["clean_rings_flagged"]}/30 |')
        lines+=[]
    lines+=['','| Method | Pores with box IoU >=0.25 | >=0.5 (primary) | >=0.75 |','|---|---:|---:|---:|']
    for method,result in post['methods'].items():
        s=result['localization_sensitivity'];lines.append(f'| {method} | {s["0.25"]}/60 | {s["0.5"]}/60 | {s["0.75"]}/60 |')
    lines+=['','Relaxing IoU changes the question being answered; the original IoU>=0.5 results remain primary. In this diagnostic, lowering minimum area does not increase the correctly localized pore count at all: CNN remains at 50/60, median at 6/60. It adds false regions. Removing small-region filtering alone therefore does not repair the missed detections. The score support and predicted shape need attention; any revised method must be developed on training/validation data and tested on fresh confirmation.',
        '', 'For the previously identified test-000 intrinsic-pore miss, the truth contains 37 pixels. The CNN threshold activates 12 of those pixels; area filtering removes them. Reducing the area limit to 1 restores them, but the best box IoU is only 0.268, so the pore still fails the unchanged localization criterion. Median similarly restores 7 pixels but reaches only IoU 0.161. This selected explanatory case demonstrates incomplete spatial support beneath the filtering symptom; it is not an independent performance estimate.',
        '', '![Known small-pore failure](figures/circular_robustness_20261004/small_pore_failure.png)',
        '', '## Current software and preserved history','',
        'The default CLI/GUI inspector now uses the v4 circular checkpoint, normalization and periodic region extraction. Historical September crop models remain explicitly selectable. The current workbench generates circular single/multi-pore examples with the current profile and raw-data contract. Research examples are not silently added to training.',
        '', 'The old frozen renderer/CV sources, datasets, models and reports are preserved. A new GUI adapter and workbench wrapper avoid editing the renderer source covered by historical hashes. The new renderer wrapper labels constant-coefficient evaluation angles separately from actual local incidence; historical metadata remain unchanged.',
        '', '## Development decisions from this evidence','',
        'Keep CNN as the primary method and median as the explicit comparison baseline. The observed median alarm increase rules out claiming centre-error robustness for its current form. A mechanism to test is radial misregistration among neighbouring windows: a centre error makes the seam move radially around the estimated ring, while the selected method uses calibration learned under nominal extraction. This is a code-based explanation to investigate, not a uniquely established causal diagnosis.',
        '', 'Prioritize training/validation experiments that improve small-pore spatial coverage and handle declared ring-coordinate errors within the existing CNN/median families. Retain the original IoU criterion and separately report clean-ring alarms. Do not fix the result by relaxing the matching criterion or selecting a threshold on this opened stress set. Binary union localization remains the task; separate overlapping-pore recovery is a documented limitation rather than an unrequested expansion to a new model family.',
        '', 'Before adopting a finer grid, extend the current component check to the complete acquisition chain at matched physical sampling, blur and noise assumptions. A changed renderer requires a new dataset/model version and fresh confirmation backgrounds. Independently crossing diameter, offset and existing profile amplitude would remove the confounding in this stress design. Unfamiliar profiles remain paused.',
        '', '## Reproduction and limits','',
        'Run `WeldDefectDetection/run_circular_robustness_20261004.py all` for the stress campaign. The protocol, source archive and model hashes were recorded before new-scene evaluation. Run `circular_robustness.investigation` through the documented project Python path for explanatory diagnostics. Read PROJECT_CURRENT.md for GUI and release commands.',
        '', 'After native-image replay, this delivery resumed scoring with two independent CPU processes to reduce elapsed time. Each retains four Torch threads and executes the unchanged frozen evaluator on disjoint backgrounds. Partial aggregates are suppressed; the original evaluator produces the final aggregate from all 168 saved scene results. execution/parallel_protocol.json records this scheduling change and the number of already completed scenes. It changes neither the scientific protocol nor the detector.',
        '', 'All findings are conditional on the same synthetic surface/response/acquisition family. They do not establish measured weld validity, automatic localization performance, physical pore interaction, or a general guarantee of robustness. Unfamiliar weld profiles remain deliberately out of scope.']
    doc.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=[doc,*figdir.glob('*.png'),c.OUT/'protocol.json',c.OUT/'protocol_native_v2.json',c.OUT/'results.json',c.OUT/'paired_localisation.json',c.OUT/'index.json',c.OUT/'native_inputs/index.json',c.OUT/'investigation/localisation_resampling.json',c.OUT/'investigation/resolution.json',c.OUT/'investigation/postprocessing.json',c.OUT/'investigation/fixed_failure_case.json']
    files.extend(c.OUT/name for name in ('README.md','audit.json','dataset_manifest.json','analysis_sources.zip','sources.zip'))
    if (c.OUT/'execution/parallel_protocol.json').exists():files.append(c.OUT/'execution/parallel_protocol.json')
    c.write(c.OUT/'delivery_manifest.json',dict(files={str(p):c.sha(p) for p in files}))
    print(doc,flush=True)

if __name__=='__main__':run()
