"""Rebuild report tables from frozen records; never train or change the campaign."""
import hashlib
import json
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1]
ROOT = REPORT.parents[1]
CAMPAIGN = ROOT / 'DataSimulation/data/output/count_diversity_20261027_v2'
RESULTS = json.loads((CAMPAIGN / 'results.json').read_text())
FROZEN = json.loads((CAMPAIGN / 'frozen_models.json').read_text())
ARMS = ['single1_24', 'single1_96', 'single4_96', 'single64_96', 'mixed_24', 'mixed_96']
COUNTS = ['1', '4', '64']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows):
    tp, fp, fn = (sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn'))
    clean = [r for r in rows if r['source_class'] == 'clean']
    return dict(tp=tp, fp=fp, fn=fn, precision=tp/max(1, tp+fp),
                recall=tp/max(1, tp+fn), f1=2*tp/max(1, 2*tp+fp+fn),
                pixel_iou=sum(r['intersection'] for r in rows)/max(1, sum(r['union'] for r in rows)),
                clean_rings=len(clean), clean_rings_flagged=sum(r['fp'] > 0 for r in clean),
                clean_false_regions=sum(r['fp'] for r in clean), scenes=len(rows))


# Independently recompute all displayed measures from per-scene outcomes.
verified = 0
for name, record in RESULTS.items():
    assert len(record['rows']) == 864
    assert len({r['background_id'] for r in record['rows']}) == 48
    for count in COUNTS:
        rows = [r for r in record['rows'] if r['count_multiplier'] == int(count)]
        actual = metrics(rows)
        assert actual['scenes'] == 288 and actual['clean_rings'] == 96
        for key, value in actual.items():
            assert abs(value-record['by_count'][count][key]) < 1e-12, (name, count, key)
        verified += 1


def selected(arm, kind):
    if kind == 'cnn':
        seed = FROZEN['arms'][arm]['cnn']['seed']
        record = RESULTS[f'{arm}_cnn_{seed}']
        assert record['selected']
        return record
    return RESULTS[f'{arm}_median']


def arm_label(arm):
    return arm.replace('_', r'\_')


def count_tex(count):
    return str(count) + r'\(\times\)'


def longtable(caption, label, spec, header, rows):
    head = ' & '.join(header) + r' \\'
    return '\n'.join([
        r'\begingroup', r'\small', r'\renewcommand{\baselinestretch}{1}\selectfont',
        r'\setlength{\tabcolsep}{3pt}', r'\renewcommand{\arraystretch}{1.25}',
        r'\begin{longtable}{' + spec + '}',
        r'\caption{' + caption + r'}\label{' + label + r'}\\',
        r'\toprule', head, r'\midrule', r'\endfirsthead',
        r'\multicolumn{' + str(len(header)) + r'}{l}{\small\tablename\ \thetable\ (continued)}\\',
        r'\toprule', head, r'\midrule', r'\endhead',
        r'\midrule', r'\multicolumn{' + str(len(header)) + r'}{r}{\small Continued on next page}\\',
        r'\endfoot', r'\bottomrule', r'\endlastfoot', *rows,
        r'\end{longtable}', r'\endgroup', ''])


tables = REPORT / 'tables'
tables.mkdir(exist_ok=True)
transfer = []
for arm in ARMS:
    bg = int(arm.split('_')[-1])
    train = r'1, 4, 64' if arm.startswith('mixed') else arm.split('_')[0].replace('single', '')
    for count in COUNTS:
        c, m = (selected(arm, kind)['by_count'][count] for kind in ('cnn', 'median'))
        transfer.append(' & '.join([train, str(bg), count, f"{c['f1']:.4f}", f"{m['f1']:.4f}",
                                    f"{c['clean_rings_flagged']}/96", f"{m['clean_rings_flagged']}/96"]) + r' \\')
    transfer.append(r'\addlinespace[4pt]')
(tables/'complete_transfer.tex').write_text(longtable(
    r'Complete development-to-confirmation comparison. Count values denote effective-count multipliers. Clean columns count acquisitions with at least one predicted region. Each three-row group uses one frozen CNN and one frozen median. \textsuperscript{[\ref{ev:28}, \ref{ev:34}, \ref{ev:44}]}',
    'tab:5-complete-transfer', '@{}p{2.0cm}p{1.6cm}p{1.6cm}rrrr@{}',
    ['Training counts', 'Training backgrounds', 'Test counts', 'CNN F1', 'Median F1', 'CNN clean', 'Median clean'], transfer), encoding='utf-8')

appendix = [r'''\chapter{Complete experimental results}\label{app:complete-results}

This appendix records the completed count-noise and training-diversity study, \texttt{count\_diversity\_20261027\_v2}. It expands the main results without introducing new training, retuning or confirmation samples. All tables derive from the frozen selections and saved per-scene confirmation records. Evidence~\ref{ev:44} supplies a compact numerical extract, source hashes and the table-generation check. \textsuperscript{[\ref{ev:28}, \ref{ev:34}, \ref{ev:44}]}

\section{How to read the experiment ledger}\label{sec:ledger-reading}

An arm name joins its fitting count coverage and number of independent training backgrounds. For example, \texttt{single4\_96} uses 4\(\times\) fitting images from 96 backgrounds, whereas \texttt{mixed\_24} uses all three count levels from 24 backgrounds. Table~\ref{tab:4-2} gives the available scene counts. Every method processes the same 48 confirmation backgrounds at each test count, with 96 clean and 192 defective acquisitions. A test count changes acquisition variability; it does not change the fitted weights or operating point. Shared normalisation nevertheless exposes all arms to the three count levels within the first 24 training backgrounds, as Section~\ref{sec:4-3} explains. \textsuperscript{[\ref{ev:04}, \ref{ev:05}, \ref{ev:29}, \ref{ev:34}]}

The selected CNN in each arm follows validation, not the largest confirmation score. Tables~\ref{tab:ledger-detection} and \ref{tab:ledger-alarms} include those six selected CNNs, the six corresponding medians and the two retained-model controls. Table~\ref{tab:all-seeds} reports all 18 CNN runs separately. The repeats share background geometry, so 288 scenes at one count level do not represent 288 independent welds. \textsuperscript{[\ref{ev:28}, \ref{ev:34}]}

\section{Frozen development choices}\label{sec:ledger-settings}

The Gaussian standard deviation belongs to median preprocessing, in pixels. Threshold denotes a standardised residual for the median and a sigmoid score for the CNN. Area denotes the minimum connected-component size in pixels; closing denotes square-kernel width. All selected CNN runs use epoch 20. The retained controls preserve their earlier normalisation and settings, as Section~\ref{sec:5-6-2} explains. \textsuperscript{[\ref{ev:28}, \ref{ev:34}]}
''']
settings = []
compact_settings = {}
for arm in ARMS:
    d = FROZEN['arms'][arm]
    compact_settings[arm] = {}
    for kind in ['cnn', 'median']:
        obj = d[kind]
        best = obj['selection']['best']
        compact_settings[arm][kind] = {k: best[k] for k in ('threshold', 'min_area', 'closing')}
        compact_settings[arm][kind].update({'seed':obj['seed'], 'epoch':obj['epoch']} if kind=='cnn' else {'sigma':obj['spec']['sigma']})
        settings.append(' & '.join([r'\texttt{'+arm_label(arm)+'}', kind.upper() if kind=='cnn' else 'Median',
                         str(obj['seed']) if kind=='cnn' else '--',
                         '--' if kind=='cnn' else f"{obj['spec']['sigma']:g}", f"{best['threshold']:g}",
                         str(best['min_area']), str(best['closing'])]) + r' \\')
    settings.append(r'\addlinespace[3pt]')
appendix.append(longtable(r'Validation-selected settings. A dash means not applicable. \textsuperscript{[\ref{ev:28}, \ref{ev:44}]}',
                         'tab:frozen-settings', '@{}llrrrrr@{}', ['Arm', 'Method', 'Seed', r'$\sigma$', 'Threshold', 'Area', 'Closing'], settings))
appendix.append(r'''\section{Detection and mask overlap}\label{sec:ledger-detection}

TP counts one-to-one matched predicted and truth regions at circular box IoU of at least 0.5. FP counts unmatched predictions, and FN counts unmatched truth regions. Each test-count row contains 192 truth regions, so TP+FN equals 192. Precision, recall and F1 use pooled region counts; pixel IoU instead pools mask intersections and unions. Section~\ref{sec:4-7} defines the complete scoring procedure. \textsuperscript{[\ref{ev:26}, \ref{ev:34}]}
''')
method_records = [(arm, kind, selected(arm, kind)) for arm in ARMS for kind in ['cnn','median']]
method_records += [('retained', kind, RESULTS['retained_'+kind]) for kind in ['cnn','median']]
detection, alarms = [], []
for arm, kind, record in method_records:
    label = r'\texttt{'+arm_label(arm)+'} '+('CNN' if kind=='cnn' else 'median')
    for count in COUNTS:
        m = record['by_count'][count]
        detection.append(' & '.join([label, count, str(m['tp']),str(m['fp']),str(m['fn']),
                         f"{m['precision']:.3f}",f"{m['recall']:.3f}",f"{m['f1']:.4f}",f"{m['pixel_iou']:.3f}"]) + r' \\')
        alarms.append(' & '.join([label, count, f"{m['clean_rings_flagged']}/96",
                        f"{100*m['clean_rings_flagged']/96:.2f}\\%",str(m['clean_false_regions'])]) + r' \\')
    detection.append(r'\addlinespace[3pt]')
    alarms.append(r'\addlinespace[3pt]')
appendix.append(longtable(r'Complete selected-model and retained-control confirmation metrics. Counts denote the test multiplier; P and R denote precision and recall. \textsuperscript{[\ref{ev:34}, \ref{ev:44}]}',
                         'tab:ledger-detection','@{}lrrrrrrrr@{}', ['Method','Counts','TP','FP','FN','P','R','F1','Pixel IoU'],detection))
appendix.append(r'''\section{Clean-image behaviour}\label{sec:ledger-alarms}

A clean acquisition contributes one flagged image if at least one predicted region survives postprocessing. It can contribute many false regions. The two counts therefore answer different questions: how often the operator would encounter an alarm-bearing clean image, and how many spurious regions the method produces. The denominator of 96 comprises two acquisitions from each of 48 backgrounds. Zero observed alarms does not establish zero alarm probability on future backgrounds. \textsuperscript{[\ref{ev:26}, \ref{ev:34}]}
''')
appendix.append(longtable(r'Clean-acquisition alarms for all selected methods and retained controls. \textsuperscript{[\ref{ev:34}, \ref{ev:44}]}',
                         'tab:ledger-alarms','@{}lr r r r@{}',['Method','Test counts','Flagged clean','Alarm rate','False regions'],alarms))
appendix.append(r'''\section{All CNN training runs}\label{sec:ledger-seeds}

Each seed controls initialisation and training sampling. Each row retains that run's validation-selected checkpoint and operating point; the mark identifies the additional validation choice among seeds for the principal comparison. These are individual-run confirmation F1 scores, not confidence intervals or independent experimental datasets. Table~\ref{tab:5-4} separately reports paired uncertainty over confirmation backgrounds. \textsuperscript{[\ref{ev:05}, \ref{ev:28}, \ref{ev:34}]}
''')
seeds=[]
for arm in ARMS:
    for c in FROZEN['arms'][arm]['cnn_candidates']:
        rec=RESULTS[f"{arm}_cnn_{c['seed']}"]
        seeds.append(' & '.join([r'\texttt{'+arm_label(arm)+'}', str(c['seed']),str(c['epoch']),
                     'Yes' if rec['selected'] else 'No']+[f"{rec['by_count'][m]['f1']:.4f}" for m in COUNTS]) + r' \\')
    seeds.append(r'\addlinespace[3pt]')
appendix.append(longtable(r'All 18 CNN runs on the same confirmation population. Selected means validation-selected within the arm. \textsuperscript{[\ref{ev:28}, \ref{ev:34}, \ref{ev:44}]}',
                         'tab:all-seeds','@{}lrrlrrr@{}',['Arm','Seed','Epoch','Selected','F1 at '+count_tex(1),'F1 at '+count_tex(4),'F1 at '+count_tex(64)],seeds))
(REPORT/'appendix_results.tex').write_text('\n'.join(appendix),encoding='utf-8')
extract={'campaign':CAMPAIGN.name, 'source_sha256':{n:sha(CAMPAIGN/n) for n in ['results.json','frozen_models.json']},
         'verified_per_count_groups':verified,'settings':compact_settings,
         'results':{name:{k:r[k] for k in ['name','arm','kind','by_count']} | {'seed':r.get('seed'),'selected':r.get('selected')} for name,r in RESULTS.items()}}
(REPORT/'evidence/results_detail_20261102.json').write_text(json.dumps(extract,indent=2),encoding='utf-8')
print(json.dumps({'verified_groups':verified,'selected_transfer_rows':18,'detailed_metric_rows':42,'clean_alarm_rows':42,'seed_rows':18,'settings_rows':12}))
