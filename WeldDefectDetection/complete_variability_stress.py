"""Complete the declared 20% stress test with explicitly flagged extrapolation.

The original 0-60 degree guard remains in place for every 0/5/10% scene.
Only a 20% stress scene that exceeds 60 degrees uses a separate, unqualified
constant-B assumption artifact. B, geometry, labels, noise, trained models and
thresholds are unchanged. The amendment is recorded before any confirmation
scores are inspected. This entry point leaves all frozen primary sources intact.
"""
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'DataSimulation'))
from bse_renderer.renderer.emission_b import load_emission_b_fit
from bse_renderer.models import WeldLayoutRuntime
from variability_campaign import campaign as c
from variability_campaign.renderer import VariableSurfaceEngine

ROOT=REPO/'DataSimulation/data/output/variability_campaign_20260926_v1'
STRESS_MODEL=ROOT/'model/stress_constant_b.json'


class StressDomainEngine(VariableSurfaceEngine):
    def _resolve_emission_b_coeff(self,normals,material_b,incidence_tilt_deg):
        angle=np.degrees(np.arccos(np.clip(normals[...,2],0,1)))
        if float(angle.max())<=60.00001:
            return super()._resolve_emission_b_coeff(normals,material_b,incidence_tilt_deg)
        if self.amplitude!=.2:
            # Do not silently expand the primary evaluation domain.
            return super()._resolve_emission_b_coeff(normals,material_b,incidence_tilt_deg)
        original=self.emission_b_fit
        try:
            self.emission_b_fit=load_emission_b_fit(STRESS_MODEL,'assumed_constant')
            value,metadata=super()._resolve_emission_b_coeff(normals,material_b,incidence_tilt_deg)
            metadata['stress_extrapolation']=dict(original_upper_deg=60.,maximum_deg=float(angle.max()),
                pixels_above_original_domain=int((angle>60).sum()),pixels=int(angle.size),physical_qualification=False)
            return value,metadata
        finally:
            self.emission_b_fit=original


def stress_engine(method='finite_difference'):
    return StressDomainEngine(image_size=c.FRAME*c.FRAMES,pixel_size_mm=c.SPACING,
        render_model_mode='formula_only',emission_b_mode='assumed_constant',emission_b_path=str(c.MODEL),
        render_view_mode='top_surface',weld_layout=WeldLayoutRuntime(mode='straight_seam'),
        frame_size=c.FRAME,halo=32,gradient_method=method)


def prepare_amendment():
    c.verify_sources(ROOT);c.require(ROOT,'train')
    path=ROOT/'stress_domain_amendment.json'
    if path.exists():
        a=c.read(path)
        if a['extension_source_sha256']!=c.sha(__file__) or a['stress_model_sha256']!=c.sha(STRESS_MODEL):
            raise ValueError('Recorded stress extension changed.')
        return
    if (ROOT/'results.json').exists():raise ValueError('Cannot introduce this amendment after viewing confirmation scores.')
    original=c.read(c.MODEL)
    model={**original,'assumed_tilt_domain_deg':[0.,90.],
        'decision':'Exploratory 20 percent geometric stress extrapolation only; original primary-domain guard unchanged.',
        'coefficient_source':{'path':str((c.MODEL.parent/original['coefficient_source']['path']).resolve()),'sha256':original['coefficient_source']['sha256']},
        'limitations':original['limitations']+['Angles above 60 degrees are mathematical extrapolation without supporting angular qualification.']}
    c.write(STRESS_MODEL,model)
    a=dict(schema='stress-domain-amendment/v1',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        trigger=c.read(ROOT/'stress_domain_failure.json'),confirmation_scores_examined=False,
        primary_levels_unchanged=[0.,.05,.1],extended_level=.2,
        primary_assumed_domain_deg=[0.,60.],stress_function_domain_deg=[0.,90.],
        B_unchanged=original['fitted_b'],labels_noise_models_thresholds_unchanged=True,
        mathematical_justification='abs(B)<1 implies 1+B*sin(alpha)*sin(phi)*cos(varphi)>0 over the outward hemisphere for all incidence angles 0..90. Positivity is not physical validation.',
        interpretation='20 percent results include explicitly flagged extrapolation; report separately from primary 0..10 percent evidence. No hard case is dropped or geometrically softened.',
        extension_source_sha256=c.sha(__file__),stress_model_sha256=c.sha(STRESS_MODEL),
        frozen_models_sha256=c.sha(ROOT/'frozen_models.json'),original_protocol_sha256=c.sha(ROOT/'protocol.json'))
    c.write(path,a)
    with zipfile.ZipFile(ROOT/'stress_extension_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.write(__file__,Path(__file__).name);z.write(STRESS_MODEL,'stress_constant_b.json')


def annotate():
    rows=c.read(ROOT/'datasets/confirmation/index.json')
    extrap=[r for r in rows if 'stress_extrapolation' in r['renderer_metadata']['emission_b']]
    summary=dict(scenes=len(extrap),backgrounds=len({r['background_id'] for r in extrap}),
        all_at_20_percent=all(r['amplitude']==.2 for r in extrap),
        maximum_deg=max([r['renderer_metadata']['emission_b']['stress_extrapolation']['maximum_deg'] for r in extrap],default=0.),
        scene_ids=[r['scene_id'] for r in extrap],primary_domain_unchanged=True)
    c.write(ROOT/'stress_extrapolation_summary.json',summary)
    assert summary['all_at_20_percent']
    a=c.read(ROOT/'stress_domain_amendment.json')
    assert c.sha(ROOT/'frozen_models.json')==a['frozen_models_sha256']
    note=(f'> **Stress-domain amendment:** The 0%, 5% and 10% primary conditions retain the original 0-60 degree assumed incidence domain. '
          f'The 20% exploratory stress condition required constant-B extrapolation in {summary["scenes"]} scenes from {summary["backgrounds"]} backgrounds, '
          f'with maximum local incidence {summary["maximum_deg"]:.3f} degrees. The coefficient, labels and trained models were unchanged. '
          'This was recorded before confirmation scoring and adds no physical validation. See `stress_domain_amendment.json`.\n\n')
    report=ROOT/'REPORT.md';text=report.read_text(encoding='utf-8')
    if not text.startswith('> **Stress-domain amendment:'):
        report.write_text(note+text,encoding='utf-8')
    c.seal(ROOT,'report',[report,*ROOT.glob('*.png'),ROOT/'stress_domain_amendment.json',ROOT/'stress_extrapolation_summary.json'])
    files={str(p.resolve()):c.sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='FINAL_MANIFEST.json'}
    c.write(ROOT/'FINAL_MANIFEST.json',dict(files=files,physical_validation=False,scope='Primary campaign plus explicitly flagged stress extrapolation'))


if __name__=='__main__':
    prepare_amendment()
    c.engine=stress_engine
    for stage in ('confirmation','evaluate','report'):c.run(ROOT,stage)
    annotate()
