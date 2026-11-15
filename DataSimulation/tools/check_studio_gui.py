"""Exercise the application's own Tk controls/event queues; writes QA evidence."""
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'DataSimulation'), str(ROOT/'WeldDefectDetection')]
from bse_renderer.gui import App


def pump(app, finished, limit=90):
    start = time.monotonic()
    while not finished():
        app.root.update()
        if time.monotonic()-start > limit:
            raise TimeoutError('Desktop event-loop test timed out.')
        time.sleep(.02)
    app.root.update()


def main():
    out = ROOT/'data/audit_circular_current_20261004/historical_legacy'
    out.mkdir(parents=True, exist_ok=True)
    app = App(ROOT/'DataSimulation/config')
    app.root.withdraw()
    checks = {}
    try:
        app.root.update()
        checks['tabs'] = [app.notebook.tab(t, 'text') for t in app.notebook.tabs()]
        assert checks['tabs'][:3] == ['Overview', 'Generate images', 'Inspect images']
        app.notebook.select(app.inspection_panel)
        app.root.update()
        # This historical smoke check intentionally covers the three legacy methods.
        app.inspection_panel.campaign.set('Historical | straight crops | 26 Sep')
        app.inspection_panel._campaign_changed()
        app.inspection_panel.example()
        app.inspection_panel.output.set(str(out/'gui_inference'))
        app.inspection_panel.start()
        pump(app, lambda: not app.inspection_panel.busy)
        result = app.inspection_panel.result
        assert result is not None
        assert len(result['records']) == 3
        assert all(r['status'] == 'complete' for r in result['records'])
        checks['gui_example_result'] = result['output']
        checks['gui_methods_complete'] = [r['method'] for r in result['records']]
        checks['preview_has_image'] = bool(app.inspection_panel.preview.cget('image'))
        assert checks['preview_has_image']
        app.v_image_size.set(32)
        app.v_num_images.set(1)
        app.v_output_dir.set(str(out/'gui_generation'))
        app.v_disable_batch_noise.set(True)
        app._generate_batch()
        pump(app, lambda: not app._batch_running)
        checks['generation_status'] = app.status_var.get()
        assert checks['generation_status'].startswith('Generated 1 images')
        generated = sorted((out/'gui_generation').glob('studio_*/manifest.json'))[-1]
        manifest = json.loads(generated.read_text())
        assert manifest[0]['metadata']['render_view_mode'] == 'top_surface'
        assert manifest[0]['metadata']['weld_geometry_mode'] == 'closed_fusion'
        assert abs(manifest[0]['metadata']['material']['b']-.920694853227) < 1e-8
        checks['gui_batch_settings_reached_export'] = True
        app.inspection_panel.start()
        pump(app, lambda: not app.inspection_panel.busy)
        generated_result = app.inspection_panel.result
        assert generated_result is not None and len(generated_result['records']) == 1
        assert generated_result['records'][0]['status'] == 'complete'
        assert generated_result['records'][0]['shape'] == [32, 32]
        checks['generate_to_cv_handoff'] = generated_result['output']
        pore = app._defect_controls['keyhole_pore']
        pore['enabled'].set(True)
        pore['size'].set(9.)
        try:
            app._manual_sample()
        except ValueError as exc:
            checks['out_of_range_pore_rejected'] = str(exc)
        else:
            raise AssertionError('Oversized pore was silently clamped.')
        # Layout geometry checks are not a screenshot or visual sign-off.
        for width, height in ((1220, 900), (1000, 760)):
            app.root.geometry(f'{width}x{height}')
            app.root.update()
            button = app.inspection_panel.run_button
            x = button.winfo_rootx()-app.root.winfo_rootx()
            y = button.winfo_rooty()-app.root.winfo_rooty()
            assert 0 <= x < width and 0 <= y < height
        checks['primary_action_in_bounds_at_two_sizes'] = True
        checks['visual_capture_status'] = 'Unavailable: desktop helper reports window-owner mismatch even after refresh.'
        (out/'gui_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
        print(json.dumps(checks))
    finally:
        app.root.destroy()


if __name__ == '__main__':
    main()
