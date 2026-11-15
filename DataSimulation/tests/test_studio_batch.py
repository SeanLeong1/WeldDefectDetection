from dataclasses import replace
from pathlib import Path
import json

from bse_renderer.config_loader import load_config_bundle
from bse_renderer.domain_randomisation.parameter_sampler import sample_parameters
from bse_renderer.domain_randomisation.batch_generator import BatchOptions, BatchGenerator


def test_selected_sample_and_surface_reach_export(tmp_path):
    bundle = load_config_bundle(Path(__file__).resolve().parents[1]/'config')
    sample = sample_parameters(bundle, seed=5, max_defects=0, material_name='304SS')
    sample = replace(sample, material=replace(sample.material, b=.920694853227),
                     beam=replace(sample.beam, e0_kev=60., accelerating_voltage_kv=60.), defects=[])
    options = BatchOptions(num_images=1, output_dir=tmp_path/'batch', image_size=32,
                           sample_override=sample, render_view_mode='top_surface',
                           weld_geometry_mode='closed_fusion', emission_b_mode='constant', disable_noise=True)
    summary = BatchGenerator(bundle, options).run()
    record = json.loads(Path(summary['records']).read_text().splitlines()[0])
    assert record['render_view_mode'] == 'top_surface'
    assert abs(record['emission_b_value']-.920694853227) < 1e-6
    assert record['noise_enabled'] is False
    manifest = json.loads(Path(summary['manifest']).read_text())
    assert manifest[0]['metadata']['material']['name'] == '304SS'
    assert manifest[0]['metadata']['beam']['e0_kev'] == 60
    assert manifest[0]['metadata']['weld_geometry_mode'] == 'closed_fusion'
    from inspection.engine import FrozenInspector, load_image, Options, LEGACY_CAMPAIGN
    raw = Path(manifest[0]['artifacts']['topo_raw'])
    item = load_image(raw, Options(mode='raw'), FrozenInspector(LEGACY_CAMPAIGN).norm)
    assert item['shape'] == [32, 32]
    assert item['pixel_size_mm'] == .05
    assert any('synthetic' in w for w in item['warnings'])
