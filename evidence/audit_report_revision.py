"""Check the active multifile report without running experiments or editing sources."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

REPORT=Path(__file__).resolve().parents[1]
ROOT=REPORT.parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def uncomment(s): return re.sub(r'(?<!\\)%[^\n]*','',s)
files={}
def visit(p):
    if p in files:return
    assert p.exists(),str(p)
    text=uncomment(p.read_text(encoding='utf-8'))
    files[p]=text
    for name in re.findall(r'\\(?:input|include)\{([^}]+)\}',text):
        child=REPORT/(name if Path(name).suffix else name+'.tex')
        # Optional personal front matter remains absent by design.
        if child.exists():visit(child)
        else:assert f'\\IfFileExists{{{name}.tex}}' in text,(p,name)
visit(REPORT/'main.tex')
text='\n'.join(files.values())
labels=re.findall(r'\\label\{([^}]+)\}',text)+re.findall(r'\\evidenceentry\{([^}]+)\}',text)
labels=[x for x in labels if '#' not in x]
duplicates=[k for k,v in Counter(labels).items() if v!=1]
refs=re.findall(r'\\(?:ref|eqref|pageref|autoref)\{([^}]+)\}',text)
citations={k.strip() for x in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}',text) for k in x.split(',')}
bib=(REPORT/'pubs.bib').read_text(encoding='utf-8')
bibkeys=re.findall(r'@\w+\s*\{\s*([^,]+),',bib)
images=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text)
assert not duplicates,duplicates
assert not set(refs)-set(labels),set(refs)-set(labels)
assert not citations-set(bibkeys),citations-set(bibkeys)
assert len(set(bibkeys))==len(bibkeys)
assert all((REPORT/x).exists() for x in images)
assert '??' not in text
assert not re.findall(r'\((?:Section|Table|Figure|Equation|Chapter)~\\(?:ref|eqref)\{[^}]+\}\.',text)
original=json.loads((REPORT/'evidence/source_manifest.json').read_text())
snapshots=[a for a in original['artifacts'] if 'snapshot' in a]
assert all(sha(REPORT/a['snapshot'])==a['sha256'] for a in snapshots)
provenance=json.loads((REPORT/'evidence/E41_chapter5_results_provenance.json.txt').read_text())
campaign_checks={}
for name,digest in provenance['source_hashes'].items():
    if Path(name).name in {'results.json','frozen_models.json','paired_comparisons.json','measured_noise.json','diagnostic_results.json','normalization.json','independent_final_audit.json','final_delivery_verification.json','noise_checks.json','method_checks.json'}:
        assert sha(ROOT/name)==digest,name
        campaign_checks[name]=digest
assert len(campaign_checks)==10
extract=json.loads((REPORT/'evidence/results_detail_20261102.json').read_text())
assert extract['verified_per_count_groups']==78
for name,digest in extract['source_sha256'].items():
    assert sha(ROOT/'DataSimulation/data/output/count_diversity_20261027_v2'/name)==digest
# Check the six existing primary-result rows against original numeric summaries.
resulttext=files[REPORT/'ThesisC/chapters/8ResultDiscussion.tex']
rows=re.findall(r'^(1|4|64)\\\(\\times\\\) & (CNN|Median) & (\d+) / (\d+) / (\d+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & (\d+)/96',resulttext,re.M)
assert len(rows)==6,len(rows)
for count,method,tp,fp,fn,p,r,f,iou,alarm in rows:
    name='mixed_96_cnn_1027001' if method=='CNN' else 'mixed_96_median'
    d=extract['results'][name]['by_count'][count]
    assert [int(tp),int(fp),int(fn),int(alarm)]==[d[k] for k in ['tp','fp','fn','clean_rings_flagged']]
    assert [p,r,f,iou]==[f'{d[k]:.3f}' for k in ['precision','recall','f1','pixel_iou']]
# Ignore mathematical I when checking first-person prose.
prose=re.sub(r'\\begin\{equation\}.*?\\end\{equation\}','',text,flags=re.S)
prose=re.sub(r'\\\(.*?\\\)','',prose,flags=re.S)
assert not re.search(r'\b(?:I|[Ww]e|[Uu]s|[Oo]ur)\b',prose)
receipt={'date':'2026-11-02','active_main':'main.tex','active_tex_files':len(files),
         'labels':len(labels),'reference_uses':len(refs),'missing_references':[],
         'duplicate_labels':[],'cited_bibliography_entries':len(citations),
         'uncited_bibliography_entries':sorted(set(bibkeys)-citations),
         'figures':len(images),'historical_evidence_snapshots_preserved':len(snapshots),
         'campaign_records_unchanged':campaign_checks,'verified_metric_groups':78,
         'existing_primary_result_rows_checked':len(rows),
         'active_source_sha256':{str(p.relative_to(REPORT)):sha(p) for p in files},
         'bibliography_sha256':sha(REPORT/'pubs.bib'),
         'scope':'Static source and numeric audit; desktop compilation has a separate tool receipt. No new experiments.'}
(REPORT/'evidence/revision_audit_20261102.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k not in ['campaign_records_unchanged','active_source_sha256']}))
