"""Generation stages and folder summaries, separate from frozen inference."""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import csv
import json
import time
import uuid
import numpy as np
from . import final_methods as engine
from .image_import import FORMATS, ImportSettings, import_image, load_raster


@dataclass(frozen=True)
class GenerationSettings:
    seed: int = 2811000023
    number: int = 1
    kind: str = 'open_intrinsic'
    crown_height: float = 3.
    crown_width: float = 2.8
    variation: float = .05
    diameter: float = .6
    depth: float = .25
    angle: float = 90.
    offset: float = 0.
    counts: int = 64
    dwell: float = .2
    random_pores: bool = False

    def validate(self):
        limits = {'seed': (0, 3000000000), 'number': (1, 100), 'crown_height': (.1, 5),
                  'crown_width': (1, 4), 'variation': (0, .1), 'diameter': (.15, 1),
                  'depth': (.05, .5), 'angle': (0, 360), 'offset': (-1.05, 1.05), 'dwell': (.1, 20)}
        for key, (low, high) in limits.items():
            value = getattr(self, key)
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f'{key.replace("_", " ")}: use {low} to {high}.')
        if self.kind not in ('clean', 'open_intrinsic', 'open_keyhole') or self.counts not in (1, 4, 64):
            raise ValueError('Choose a listed pore type and acquisition level.')
        if int(self.number) != self.number or int(self.seed) != self.seed:
            raise ValueError('Seed and image count must be whole numbers.')
        if self.seed + self.number - 1 > 3000000000:
            raise ValueError('Batch seeds exceed the supported range.')


def generate_scene(settings, folder):
    """Retain the actual intermediate arrays; never modify campaign sources."""
    from raised_crown import rendering as render
    from count_study import campaign
    from bse_renderer.noise.raster_acquisition import acquire
    from bse_renderer.renderer.image_formation import form_channels
    settings.validate()

    class ConfigurableEngine(render.Engine):
        def volume(self, **kwargs):
            volume = super().volume(**kwargs)
            self.geometry_cache = render.geo.height(self.u * 2.8 / settings.crown_width,
                self.theta, self.sample_seed, self.amplitude) * (settings.crown_height / 3.)
            volume.relief_mm[0] = self.geometry_cache
            volume.fusion_mask[0] = np.abs(self.u) <= settings.crown_width / 2
            return volume

    sample, _ = render.oldg.baseline.c.sampled(render.old.load_config_bundle(engine.ROOT/'DataSimulation/config'), settings.seed, 0)
    plan = campaign.pore_plan(settings.seed, 1, settings.kind)
    if plan and not settings.random_pores:
        plan[0]['params'].update(diameter_mm=settings.diameter, depth_mm=settings.depth,
                                angle_rad=np.deg2rad(settings.angle), radial_offset_mm=settings.offset)
    sample = replace(sample, beam=replace(sample.beam, dwell_time_us=settings.dwell), defects=plan)
    renderer = ConfigurableEngine(settings.seed, settings.variation)
    arrays, metadata = render.render(renderer, sample)
    profile = replace(renderer.acquisition_profile,
        effective_count_rate=renderer.acquisition_profile.effective_count_rate * settings.counts)
    measured, receipt = acquire(renderer.acquisition_quiet_quadrants, sample.noise,
                                seed=settings.seed, dwell_us=settings.dwell, profile=profile)
    gain = float(np.interp(sample.beam.focusing_current_norm, [0, 1], [1.10, .92]))
    channels = form_channels(measured)
    stages = dict(healthy_height=render.extract(renderer.geometry_cache).astype('f4'),
                  height=arrays['height'], mask=arrays['mask'], quiet=arrays['quiet'])
    for key in ('topo', 'tx', 'ty', 'compo'):
        stages[key] = (render.extract(getattr(channels, key))*gain).astype('f4')
    stages['detectors'] = np.stack([render.extract(measured[..., i]) for i in range(4)]).astype('f4')
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    np.save(folder/'topo.npy', stages['topo'])
    np.save(folder/'topo_mask.npy', stages['mask'])
    np.savez_compressed(folder/'stages.npz', **stages)
    geometry = metadata['geometry']
    geometry.update(crown_height_mm=settings.crown_height, crown_width_mm=settings.crown_width,
                    fusion_surface_width_mm=settings.crown_width)
    metadata['renderer_metadata'].update(ebw_reinforcement_mm=settings.crown_height, ebw_width_mm=settings.crown_width)
    engine.study.write(folder/'scene.json', dict(settings=asdict(settings), parameters=asdict(sample),
        metadata=metadata, acquisition=receipt, research_split=False, physical_validation=False,
        geometry_note='Configurable surface crown; no new transport calibration or fusion-volume simulation.',
        channel_order='Channels formed on the native Cartesian raster, then interpolated to the annulus. TOPO magnitude is formed before interpolation.',
        trained_geometry=(settings.crown_height == 3 and settings.crown_width == 2.8)))
    from matplotlib.figure import Figure
    previews=folder/'previews';previews.mkdir()
    for stage in ('3D surface','Weld profile','Pores and truth','Detector channels','Directional signals','TOPO'):
        figure=Figure(figsize=(10,6),dpi=100)
        draw_stage(figure,folder/'topo.npy',stage)
        figure.savefig(previews/(stage.replace(' ','_')+'.png'),dpi=150)
    return folder/'topo.npy'


def new_folder(category):
    folder = engine.OUTPUT/category/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    return folder


def discover(folder, recursive=True):
    """Snapshot only inputs, excluding known generated auxiliaries and outputs."""
    folder = Path(folder)
    files = folder.rglob('*') if recursive else folder.iterdir()
    supported = FORMATS | {'.npy', '.npz'}
    return sorted(p for p in files if p.is_file() and p.suffix.lower() in supported
        and not p.stem.lower().endswith(('_mask', '_score'))
        and p.name.lower() not in {'stages.npz', 'predictions.npz', 'inspection.png'}
        and not any(part in {'results', 'batch_results', 'imports', 'previews'} for part in p.relative_to(folder).parts[:-1]))


def read_truth(source, input_root, mask_root=None):
    """Opt-in evaluation after inference. Masks use scored annulus coordinates."""
    source = Path(source)
    relative = source.relative_to(input_root) if input_root else Path(source.name)
    target = (Path(mask_root)/relative if mask_root else source).with_name(source.stem+'_mask.npy')
    if not target.exists():
        return None
    if target.stat().st_size > engine.MAX_BYTES:
        raise ValueError('Reference mask exceeds 256 MiB.')
    mask = np.load(target, allow_pickle=False)
    if mask.shape == (160, 5024):
        mask = mask[16:144]
    if mask.shape != (128, 5024) or not np.isfinite(mask).all() or not np.isin(mask, [0, 1, 255]).all():
        raise ValueError('Reference mask must be binary, 128 x 5024 or 160 x 5024, in prepared annulus coordinates.')
    return mask > 0


def evaluate_result(result, mask):
    truth = engine.metrics.regions(mask)[1]
    return {m: engine.metrics.match(boxes, truth, 5024) for m, boxes in result['boxes'].items()}


def summarise(rows):
    good = [r for r in rows if r['status'] == 'ok']
    output = dict(total=len(rows), completed=len(good), errors=sum(r['status']=='error' for r in rows),
                  seconds=sum(r.get('seconds', 0) for r in good), methods={})
    for method in ('cnn', 'median'):
        labelled = [r['evaluation'][method] for r in good if method in r.get('evaluation', {})]
        count = sum(r[method+'_candidates'] for r in good)
        stats = dict(candidates=count, flagged_images=sum(r[method+'_candidates'] > 0 for r in good),
                     mean_candidates=count/len(good) if good else None, labelled_images=len(labelled))
        if labelled:
            tp, fp, fn = [sum(r[k] for r in labelled) for k in ('tp', 'fp', 'fn')]
            stats.update(tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else None,
                         recall=tp/(tp+fn) if tp+fn else None, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
        output['methods'][method] = stats
    return output


def inspect_folder(paths, input_root, inspector, output, progress, cancelled,
                   raster_template=None, use_masks=False, mask_root=None):
    rows=[];output=Path(output)
    for i, source in enumerate(paths):
        if cancelled():break
        row=dict(source=str(source), status='error', error='', evaluation={})
        try:
            prepared=source
            if source.suffix.lower() in FORMATS:
                if raster_template is None:raise ValueError('Raster registration has not been prepared.')
                settings=ImportSettings(**raster_template['settings'])
                _, _, info=load_raster(source, settings.frame)
                if (info['width'], info['height']) != (raster_template['image']['width'], raster_template['image']['height']):
                    raise ValueError('Dimensions differ from the batch registration. Prepare this image separately.')
                prepared, _=import_image(source, settings, inspector.norm, output)
            result=inspector.infer(prepared)
            saved=engine.export_result(result, output/f'{i+1:04d}')
            label_source=Path(result.get('import_metadata',{}).get('source',source))
            row.update(status='ok', seconds=result['seconds'], cnn_candidates=len(result['boxes']['cnn']),
                       median_candidates=len(result['boxes']['median']), output=str(saved), prepared=str(prepared),source=str(label_source))
            if use_masks:
                try:
                    mask=read_truth(label_source, input_root, mask_root)
                    if mask is not None:row['evaluation']=evaluate_result(result, mask)
                    else:row['label_note']='No matching reference mask'
                except Exception as exc:row['label_note']=str(exc)
        except Exception as exc:row['error']=str(exc)
        rows.append(row);progress(i+1, len(paths), row)
        save_summary(output, rows, len(paths), cancelled())
    return save_summary(output, rows, len(paths), cancelled()), rows


def save_summary(output, rows, requested, cancelled):
    summary=summarise(rows)
    summary.update(requested=requested, cancelled=bool(cancelled), remaining=requested-len(rows),
        matching='One-to-one circular box IoU >= 0.5; connected components of binary reference mask',
        reference_policy='Optional external masks in prepared annulus coordinates; never used in inference',
        physical_validation=False)
    engine.study.write(Path(output)/'summary.json', dict(summary=summary, images=rows))
    evaluation_keys=[f'{m}_{k}' for m in ('cnn','median') for k in ('tp','fp','fn')]
    keys=['source','status','cnn_candidates','median_candidates','seconds',*evaluation_keys,'error','label_note','output']
    with (Path(output)/'images.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f, fieldnames=keys, extrasaction='ignore');writer.writeheader()
        for row in rows:
            expanded={**row,**{f'{m}_{k}':v for m,scores in row.get('evaluation',{}).items() for k,v in scores.items()}}
            writer.writerow(expanded)
    return summary


def draw_stage(figure, path, stage):
    figure.clear()
    with np.load(Path(path).parent/'stages.npz', allow_pickle=False) as file:
        arrays={k:file[k] for k in file.files}
    angle=np.arange(5024)*360/5024;radial=(np.arange(160)-79.5)*.05
    col=int(np.argmax(arrays['mask'].sum(axis=0))) if arrays['mask'].any() else 1256
    columns=(np.arange(80)+col-40)%5024
    crop_extent=(-2,2,-4,4)
    if stage == '3D surface':
        ax=figure.add_subplot(121, projection='3d')
        th=np.deg2rad(angle[::16]);rr=40+radial[::3]
        ax.plot_surface(rr[:,None]*np.cos(th), rr[:,None]*np.sin(th), arrays['height'][::3,::16],
                        cmap='viridis', linewidth=0, rcount=54, ccount=314)
        ax.set(xlabel='X (mm)',ylabel='Y (mm)',zlabel='Height (mm)')
        ax.set_title('Complete weld (height axis enlarged)',fontsize=10)
        ax.set_box_aspect((1,1,.35))
        local=figure.add_subplot(122,projection='3d')
        arc=(np.arange(80)-40)*.05
        xx,yy=np.meshgrid(arc,radial)
        local.plot_surface(xx,yy,arrays['height'][:,columns],cmap='viridis',linewidth=0,rcount=160,ccount=80)
        local.set(xlabel='Arc offset (mm)',ylabel='Radial offset (mm)',zlabel='Height (mm)')
        local.set_title('Magnified pore neighbourhood' if arrays['mask'].any() else 'Magnified surface neighbourhood',fontsize=10)
        local.set_box_aspect((1,1.5,.8))
        from matplotlib.ticker import FormatStrFormatter, MaxNLocator
        for axes in (ax,local):
            axes.tick_params(labelsize=8)
            axes.zaxis.set_major_locator(MaxNLocator(4));axes.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    elif stage == 'Detector channels':
        low,high=np.percentile(arrays['detectors'],[1,99])
        for i,ax in enumerate(figure.subplots(2,2).flat):
            ax.imshow(arrays['detectors'][i],aspect='auto',origin='lower',extent=(0,360,-4,4),cmap='gray',vmin=low,vmax=high)
            ax.set(title=f'Detector {i+1}',xlabel='Angle (degrees)',ylabel='Radial offset (mm)')
    else:
        keys={'Weld profile':['healthy_height'], 'Pores and truth':['height','mask'],
              'TOPO':['topo'], 'Directional signals':['tx','ty','compo']}
        chosen=keys[stage]
        axes=np.atleast_1d(figure.subplots(len(chosen),1))
        for key,ax in zip(chosen,axes):
            cm='viridis' if 'height' in key else 'gray'
            local=stage=='Pores and truth'
            values=arrays['healthy_height']-arrays['height'] if local and key=='height' else arrays[key]
            im=ax.imshow(values[:,columns] if local else values,aspect='auto',origin='lower',extent=crop_extent if local else (0,360,-4,4),cmap=cm)
            ax.set(title={'healthy_height':'Crown before pore injection', 'height':'Injected surface depression',
                'mask':'Geometric pore mask', 'topo':'TOPO contrast magnitude', 'tx':'Signed X contrast',
                'ty':'Signed Y contrast', 'compo':'Sum of detector signals'}[key],xlabel='Arc offset from pore (mm)' if local else 'Angle (degrees)',ylabel='Radial offset (mm)')
            figure.colorbar(im,ax=ax,shrink=.8,label='Height (mm)' if 'height' in key else 'Value')
    figure.tight_layout(pad=2)
    if stage=='3D surface':figure.subplots_adjust(left=.01,right=.94,wspace=.32,bottom=.10,top=.90)
