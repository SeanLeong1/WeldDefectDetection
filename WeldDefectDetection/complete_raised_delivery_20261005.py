"""Complete post-freeze diagnostics, teaching, GUI validation and release."""
from pathlib import Path
import sys,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'DataSimulation'),str(ROOT/'WeldDefectDetection')]

if __name__=='__main__':
    from raised_crown import campaign as c
    from raised_crown.learning import verify_freeze
    verify_freeze();c.audit()
    assert (c.OUT/'results.json').exists() and (c.OUT/'robustness_results.json').exists()
    from raised_crown.small_pores import run as small
    from raised_crown.sampling_signal import run as sampling
    from raised_crown.trace_sample import run as trace
    from raised_crown.selection_diagnostic import run as selection
    from raised_crown.dwell_diagnostic import run as dwell
    if not (c.OUT/'small_pore_diagnostic.json').exists():small()
    if not (c.OUT/'small_pore_signal_sampling.json').exists():sampling()
    if not (c.OUT/'traced_sample.json').exists():trace()
    selection()
    dwell()
    from raised_crown.reporting import report
    report()
    subprocess.run([sys.executable,str(ROOT/'DataSimulation/tools/check_raised_crown_gui.py')],cwd=ROOT,check=True)
    from raised_crown.publish import run as publish
    publish()
    sources=list((ROOT/'WeldDefectDetection/raised_crown').glob('*.py'))+[Path(__file__),ROOT/'DataSimulation/tools/package_raised_crown.py',ROOT/'DataSimulation/tools/check_raised_crown_gui.py']
    c.write(c.OUT/'analysis_sources.json',dict(files={str(p.relative_to(ROOT)):c.sha(p) for p in sources},
        status='Post-freeze descriptive analyses and delivery code; original prospective scientific source hashes remain in protocol.json and cv_protocol.json'))
    with zipfile.ZipFile(c.OUT/'analysis_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,p.relative_to(ROOT))
    subprocess.run([sys.executable,str(ROOT/'DataSimulation/tools/package_raised_crown.py')],cwd=ROOT,check=True)
    c.audit();verify_freeze()
    records=[p for p in c.OUT.glob('*.json') if p.name!='delivery_manifest.json']
    c.write(c.OUT/'delivery_manifest.json',dict(files={p.name:c.sha(p) for p in records},
        complete=True,historical_evidence_preserved=True,physical_validation=False,electron_depth_cutoff_implemented=False))
    print('COMPLETE: research, diagnostics, GUI, preserved history and relocated release',flush=True)
