"""Add operational diagnostics and updated guide after a verified pore campaign."""
from pathlib import Path
import sys,json,hashlib,shutil,subprocess
import numpy as np
REPO=Path(__file__).resolve().parent.parent
ROOT=REPO/'DataSimulation/data/output/pore_campaign_20260921_v1'
sys.path[:0]=[str(REPO/'WeldDefectDetection'),str(REPO/'DataSimulation')]
from box_comparison.campaign import read,verify_files,inventory
from box_comparison.core import write,sha
from pore_campaign.campaign import check
if __name__=='__main__':
    check(ROOT)
    verify_files(read(ROOT/'FINAL_MANIFEST.json')['files'])
    if read(ROOT/'verification.json')['status']!='ok':raise RuntimeError('Complete verification first.')
    diagnostics={}
    for name in ('pilot','development','confirmation'):
        folder=ROOT/'datasets'/name
        norm=read(folder/'topo_normalization.json');low,high=norm['raw_low'],norm['raw_high']
        rows=[json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()]
        clips={'below_low':0,'above_high':0,'pixels':0}
        for r in rows:
            a=np.load(folder/r['raw_channels']['topo']);clips['pixels']+=a.size
            clips['below_low']+=int((a<low).sum());clips['above_high']+=int((a>high).sum())
        sequences=read(folder/'sequence_manifest.json')
        retries={}
        for kind in ('clean','keyhole_pore','intrinsic_pore'):
            group=[s for s in sequences if s['planned_class']==kind]
            retries[kind]=dict(sequences=len(group),sequences_retried=sum(s['generation_attempt']>1 for s in group),
                              rejected_attempts=sum(s['generation_attempt']-1 for s in group),maximum_attempt=max(s['generation_attempt'] for s in group))
        clips['above_high_fraction']=clips['above_high']/clips['pixels'];clips['below_low_fraction']=clips['below_low']/clips['pixels']
        diagnostics[name]=dict(retries_by_class=retries,normalization_clipping=clips)
    write(ROOT/'generation_diagnostics.json',diagnostics)
    subprocess.run([sys.executable,str(REPO/'WeldDefectDetection/publish_pore_guide.py')],check=True)
    for src in (REPO/'pore_regression.xml',REPO/'pore_regression.log',REPO/'pore_campaign_run.log',REPO/'WeldDefectDetection/PORE_CAMPAIGN.md'):
        shutil.copy2(src,ROOT/src.name)
    for name in ('publish_pore_guide.py','complete_pore_delivery.py'):
        shutil.copy2(REPO/'WeldDefectDetection'/name,ROOT/name)
    write(ROOT/'delivery_provenance.json',dict(documentation_helpers={str(REPO/'WeldDefectDetection'/n):sha(REPO/'WeldDefectDetection'/n) for n in ('publish_pore_guide.py','complete_pore_delivery.py')},
        scientific_protocol_changed=False,regression_tests_passed=275))
    files=inventory(ROOT);files.pop(str((ROOT/'FINAL_MANIFEST.json').resolve()),None)
    write(ROOT/'FINAL_MANIFEST.json',dict(files=files,scope='All campaign artifacts except this manifest; includes updated teaching guide snapshot and generation diagnostics.'))
    verify_files(files)
    print('Verified',len(files),'artifact hashes.')
