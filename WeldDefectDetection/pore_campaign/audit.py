"""Counterfactual rendering and signal diagnostics, without filtering labels."""
from dataclasses import replace
from pathlib import Path
import json
import cv2
import numpy as np
from box_comparison.core import write
from box_comparison.campaign import read
from box_comparison.inference import normalize_raw


def replay(dataset, output, config, emission, *, counterfactual=False):
    from bse_renderer.config_loader import load_config_bundle
    from bse_renderer.domain_randomisation.parameter_sampler import sample_parameters
    from bse_renderer.models import WeldLayoutRuntime
    from bse_renderer.renderer.scan_engine import ScanEngine
    from bse_renderer.validation.visibility_pilot import _quiet_sample

    class AuditEngine(ScanEngine):
        def _top_surface_section(self, volume):
            section, defects = super()._top_surface_section(volume)
            self.last_relief = section.relief_mm_map.copy()
            return section, defects

    dataset, output = Path(dataset), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(s) for s in (dataset/'samples.jsonl').read_text().splitlines()]
    groups = {}
    for r in records:
        groups.setdefault(r['sequence_id'], []).append(r)
    cfg = load_config_bundle(config)
    rows = []
    for sid, frames in groups.items():
        frames.sort(key=lambda r:r['frame_index'])
        r = frames[0]
        values = r['realized_non_noise_parameters']
        sample = sample_parameters(cfg, seed=r['seed'], max_defects=0)
        sample = replace(sample, **{n:replace(getattr(sample,n), **values[n]) for n in ('beam','material','detector','slice')},
                         defects=values['defects'], randomize_defect_location=values['randomize_defect_location'])
        sample = _quiet_sample(sample)
        engine = AuditEngine(image_size=r['source_scene_size_px'], pixel_size_mm=r['pixel_size_mm'],
            render_model_mode='formula_only', emission_b_mode='assumed_constant', emission_b_path=str(emission),
            render_view_mode='top_surface', weld_layout=WeldLayoutRuntime(**r['renderer_parameters']['weld_layout']))
        on = engine.run(sample)
        height_on = engine.last_relief.copy()
        off = engine.run(replace(sample,defects=[])) if counterfactual else None
        height_off = engine.last_relief.copy() if counterfactual else None
        arrays = {'raw':[], 'mask':[], 'sample_ids':[]}
        if counterfactual:
            arrays.update(off=[],height_delta=[])
        for r in frames:
            y,x = r['crop_y0_px'],r['crop_x0_px']
            cut = np.s_[y:y+64,x:x+64]
            quiet = on.channels.topo[cut]
            mask = on.defect_mask[cut]>0
            saved = cv2.imread(str(dataset/r['mask_path']),cv2.IMREAD_UNCHANGED)>0
            if not np.array_equal(mask,saved):
                raise ValueError('Noiseless replay changed geometry or mask.')
            if not np.isfinite(quiet).all():
                raise ValueError('Nonfinite noiseless output.')
            arrays['raw'].append(quiet); arrays['mask'].append(mask);arrays['sample_ids'].append(r['sample_id'])
            if counterfactual:
                arrays['off'].append(off.channels.topo[cut])
                arrays['height_delta'].append((height_on-height_off)[cut])
                if mask.any() and 2 <= r['frame_index'] <= 6:
                    delta = np.abs(quiet-off.channels.topo[cut])
                    noisy = np.load(dataset/r['raw_channels']['topo'])
                    sigma = max(1e-8,float(np.std(noisy-quiet)))
                    rows.append(dict(sequence_id=sid,sample_id=r['sample_id'],kind=sample.defects[0]['type'],
                        parameters=sample.defects[0]['params'],noise_std=sigma,mask_pixels=int(mask.sum()),
                        median_signal_change=float(np.median(delta[mask])),
                        fraction_above_2_noise_std=float(np.mean(delta[mask]>2*sigma)),
                        max_height_change_mm=float(np.max(np.abs((height_on-height_off)[cut]))),
                        mask_reproduced=True))
        np.savez_compressed(output/(sid+'.npz'),**{k:np.asarray(v) for k,v in arrays.items()})
        print(f'Paired replay: {sid} counterfactual={counterfactual}',flush=True)
    report=dict(status='ok',sequences=len(groups),all_masks_reproduced=True,rows=rows,
        rule='Every case retained. Signal > 2 noise standard deviations is descriptive, never a rejection rule.',
        physical_validation=False)
    write(output/'report.json',report)
    return report


def quiet_data(data, paired, normalization):
    raw = {}
    for path in sorted(Path(paired).glob('sequence-*.npz')):
        with np.load(path,allow_pickle=False) as a:
            raw.update({str(s):im for s,im in zip(a['sample_ids'],a['raw'])})
    target = np.stack([normalize_raw(raw[m['sample_id']],normalization) for m in data['meta']])
    context = np.stack([np.stack([normalize_raw(raw[s],normalization) for s in m['context_sample_ids']]) for m in data['meta']])
    return {**data,'target':target,'context':context,'reference':np.median(context,axis=1)}


def pilot_figures(dataset, paired, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dataset,paired,output=map(Path,(dataset,paired,output))
    records={r['sample_id']:r for r in [json.loads(s) for s in (dataset/'samples.jsonl').read_text().splitlines()]}
    rows=read(paired/'report.json')['rows']
    for kind in ('keyhole_pore','intrinsic_pore'):
        selected=[r for r in rows if r['kind']==kind]
        fig,axes=plt.subplots(len(selected),5,figsize=(12,2.1*len(selected)),squeeze=False)
        for ax,row in zip(axes,selected):
            r=records[row['sample_id']]
            with np.load(paired/(r['sequence_id']+'.npz')) as a:
                i=list(a['sample_ids']).index(row['sample_id'])
                noisy=np.load(dataset/r['raw_channels']['topo'])
                on,off,mask=a['raw'][i],a['off'][i],a['mask'][i]
                lo,hi=float(min(noisy.min(),on.min(),off.min())),float(max(noisy.max(),on.max(),off.max()))
                images=(off,on,noisy,abs(on-off),mask)
                for j,im in enumerate(images):
                    ax[j].imshow(im,cmap='gray' if j<3 else 'magma',**(dict(vmin=lo,vmax=hi) if j<3 else {}))
                    ax[j].set_xticks([]);ax[j].set_yticks([])
                ax[0].set_ylabel(f"d={row['parameters']['diameter_mm']:.3f}\nz={row['parameters']['depth_mm']:.3f} mm")
        for a,t in zip(axes[0],('No defect','Noiseless pore','Noisy pore','Absolute change*','Geometric mask')):a.set_title(t)
        fig.suptitle(kind+'; *difference panels individually scaled; all pilot cases retained',fontsize=11)
        fig.tight_layout();fig.savefig(output/(kind+'_pilot.png'),dpi=120);plt.close(fig)
