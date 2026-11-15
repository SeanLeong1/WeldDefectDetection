"""Carry numerical checks across the documented split-only v3 to v4 correction.

Verify archived source bytes first; geometry and the diagnostic index-zero plan
must be identical. This does not reuse any preliminary dataset for CV training.
"""
import hashlib,shutil,zipfile
from pathlib import Path
from . import generation as g,geometry
from box_comparison.core import sha


def run():
    previous=g.ROOT/'DataSimulation/data/output/circular_study_20261004_v3'
    with zipfile.ZipFile(previous/'renderer_sources.zip') as archive:
        oldgeometry=archive.read('WeldDefectDetection/circular_study/geometry.py')
        assert hashlib.sha256(oldgeometry).hexdigest()==sha(geometry.__file__)
        oldgeneration=archive.read('WeldDefectDetection/circular_study/generation.py').decode('utf-8')
        expected=oldgeneration.replace('20261004_v3','20261004_v4').replace('circular-thesis-study/v3','circular-thesis-study/v4').replace('(index//9)%4','index%4')
        assert expected==Path(g.__file__).read_text(encoding='utf-8'),'Unexpected renderer changes'
    oldplan=oldgeneration[oldgeneration.index('def plan('):oldgeneration.index('def generate(')]
    namespace=dict(np=g.np);exec(oldplan,namespace)
    for kind in g.baseline.c.KINDS:
        assert namespace['plan'](kind,2230400100,0)==g.plan(kind,2230400100,0)
    destination=g.OUT/'diagnostics';destination.mkdir(exist_ok=True)
    files={}
    for name in ('component_arrays.npz','component_checks.json','roi_equivalence.json'):
        src=previous/'diagnostics'/name;dst=destination/name
        shutil.copyfile(src,dst);assert sha(src)==sha(dst);files[name]=sha(dst)
    assert g.read(destination/'roi_equivalence.json')['status']=='passed'
    g.write(destination/'reuse_provenance.json',dict(origin=str(previous),
        reason='v4 only balances split angle strata; numerical geometry source and diagnostic seed/index-zero pore plan are identical',
        geometry_sha256=sha(geometry.__file__),files=files,preliminary_training_data_reused=False))


if __name__=='__main__':run()
