"""Label-free inference with the thesis-selected CNN and Median, without refitting."""
from pathlib import Path
from dataclasses import replace, asdict
import json, time, uuid, zipfile
import numpy as np
from scipy.ndimage import gaussian_filter
from count_study import learning as learn
from raised_crown import learning as base
from circular_study import metrics
import median_localisation_20261110 as study

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'data/final_method_inspection'
MAX_BYTES = 256 * 1024 * 1024


def load_raw(path):
    """Only read image channels. Never read a geometric mask or truth boxes."""
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('Input exceeds 256 MiB.')
    if path.suffix.lower() == '.npy':
        raw = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == '.npz':
        with zipfile.ZipFile(path) as archive:
            if sum(x.file_size for x in archive.infolist()) > MAX_BYTES:
                raise ValueError('Expanded input exceeds 256 MiB.')
        with np.load(path, allow_pickle=False) as arrays:
            key = next((k for k in ('raw', 'topo') if k in arrays), None)
            if key is None:
                raise ValueError('NPZ must contain raw or topo image values.')
            raw = arrays[key].copy()
    else:
        raise ValueError('Choose a raw TOPO .npy or .npz annulus. PNG/JPEG displays need calibrated ring extraction first.')
    if raw.ndim == 3 and raw.shape[0] in (1, 3):
        raw = raw[0]
    if raw.shape != (160, 5024) or raw.dtype.kind not in 'fiu' or not np.isfinite(raw).all():
        raise ValueError('Expected a finite 160 x 5024 raw TOPO annulus, including the radial halo.')
    if np.ptp(raw) <= 0:
        raise ValueError('TOPO image has no intensity variation.')
    return raw


class FinalInspector:
    def __init__(self):
        import torch
        torch.set_num_threads(4)
        frozen = study.verify_frozen()
        self.cnn = frozen['original']['cnn']
        self.median = frozen['selection']['combined']
        self.norm = study.read(study.OLD/'normalization.json')
        sigma = self.median['base']['sigma']
        self.calibration = study.read(study.OUT/f'development/calibration_sigma{sigma:g}.json')
        self.model = None
        self.identity = dict(cnn_arm=self.cnn['arm'], cnn_seed=self.cnn['seed'], cnn_epoch=self.cnn['epoch'],
            checkpoint_sha256=self.cnn['checkpoint_sha256'],
            frozen_selection_sha256=study.sha(study.OUT/'frozen.json'),
            normalization_sha256=study.sha(study.OLD/'normalization.json'),
            calibration_sha256=study.sha(study.OUT/f'development/calibration_sigma{sigma:g}.json'),
            cnn_operating_point={k:self.cnn['selection']['best'][k] for k in ('threshold','min_area','closing')},
            median_operating_point={k:self.median['base'][k] for k in ('sigma','threshold','min_area','closing')},
            median_box_correction=self.median['correction'])

    def infer(self, path, methods=('cnn', 'median')):
        if not methods or any(m not in ('cnn','median') for m in methods):
            raise ValueError('Select CNN, Median, or both.')
        raw = load_raw(path)
        result = dict(raw=raw, source=str(Path(path).resolve()), source_sha256=study.sha(path),
                      scores={}, masks={}, boxes={}, identity=self.identity, labels_used=False)
        sidecar=Path(path).with_suffix('.import.json')
        if sidecar.exists():
            imported=json.loads(sidecar.read_text())
            if imported.get('schema')!='weld-raster-import/v1' or imported.get('prepared_sha256')!=result['source_sha256']:
                raise ValueError('Imported image metadata does not match the prepared TOPO.')
            result['import_metadata']=imported
        start = time.monotonic()
        for method in methods:
            if method == 'cnn':
                if self.model is None:
                    self.model = base.model_for(self.cnn)
                score = base.predict_image(base.normalize(raw,self.norm),self.model)
                op = self.cnn['selection']['best']
            else:
                op = self.median['base']
                smooth = gaussian_filter(raw,op['sigma'],mode=('reflect','wrap'))
                score = learn.median_score(base.normalize(smooth,self.norm),self.calibration)
            mask, boxes = metrics.binary(score,op['threshold'],op['min_area'],op['closing'])
            if method == 'median':
                boxes = study.transform(boxes,self.median['correction'])
                mask = study.box_mask(boxes)
            result['scores'][method]=score
            result['masks'][method]=mask
            result['boxes'][method]=boxes
        result['seconds']=time.monotonic()-start
        return result


def example_path():
    rows=study.read(study.OUT/'datasets/test/index.json')
    return Path(next(r['raw_path'] for r in rows if r['scene_id']=='test-023-open_intrinsic-c64-r0'))


def generate(seed, kind, counts, dwell):
    """Generate a separate demo with the study's renderer and count acquisition."""
    from count_study import campaign as campaign
    from raised_crown import rendering as render
    from bse_renderer.noise.raster_acquisition import acquire
    from bse_renderer.renderer.image_formation import form_channels
    if not 0 <= seed <= 3_000_000_000 or kind not in campaign.KINDS or counts not in (1,4,64) or not .1 <= dwell <= 20:
        raise ValueError('Use seed 0..3000000000, counts 1/4/64, and dwell 0.1..20 us.')
    sample,_=render.oldg.baseline.c.sampled(render.old.load_config_bundle(ROOT/'DataSimulation/config'),seed,0)
    sample=replace(sample,beam=replace(sample.beam,dwell_time_us=dwell),defects=campaign.pore_plan(seed,0,kind))
    engine=render.Engine(seed,.05)
    arrays,metadata=render.render(engine,sample)
    profile=replace(engine.acquisition_profile,effective_count_rate=engine.acquisition_profile.effective_count_rate*counts)
    measured,receipt=acquire(engine.acquisition_quiet_quadrants,sample.noise,seed=seed,dwell_us=dwell,profile=profile)
    gain=float(np.interp(sample.beam.focusing_current_norm,[0,1],[1.10,.92]))
    raw=render.extract(form_channels(measured).topo*gain).astype('f4')
    folder=OUTPUT/'generated'/uuid.uuid4().hex[:12];folder.mkdir(parents=True)
    path=folder/'topo.npy';np.save(path,raw)
    study.write(folder/'scene.json',dict(metadata=metadata,parameters=asdict(sample),count_multiplier=counts,
        acquisition=receipt,research_split=False,source_sha256=study.sha(path)))
    return path


def draw_result(figure, result):
    from scipy.ndimage import map_coordinates
    from matplotlib.patches import Rectangle, Circle, ConnectionPatch
    figure.clear()
    raw=result['raw'];view=raw[16:144]
    low,high=np.percentile(raw,[1,99])
    cmap=__import__('matplotlib').colormaps['gray'].copy();cmap.set_bad('white')
    axis=np.linspace(-45,45,850);xx,yy=np.meshgrid(axis,axis)
    rr=(np.hypot(xx,yy)-40)/.05+79.5;cc=np.mod(np.arctan2(yy,xx),2*np.pi)*5024/(2*np.pi)
    ring=map_coordinates(raw,[np.clip(rr,0,159),cc],order=1,mode='grid-wrap')
    ring=np.ma.array(ring,mask=(rr<0)|(rr>159))
    imported=result.get('import_metadata')
    original=None
    if imported and imported['settings']['layout']=='circular':
        from .image_import import load_raster,sha
        path=Path(imported['source'])
        if path.exists() and sha(path)==imported['source_sha256']:
            original=load_raster(path,imported['settings']['frame'])[0]
    ax=figure.add_subplot(121)
    if original is not None:
        lim=np.percentile(original,[1,99]);ax.imshow(original,cmap='gray',vmin=lim[0],vmax=lim[1])
    else:ax.imshow(ring,origin='lower',extent=(-45,45,-45,45),cmap=cmap,vmin=low,vmax=high)
    def position(angle,radius):
        if original is None:return radius*np.cos(angle),radius*np.sin(angle)
        settings=imported['settings']
        if settings['reverse_rows']:radius=80-radius
        angle=np.deg2rad(settings['start_angle_deg'])+(-1 if settings['clockwise'] else 1)*angle
        return settings['centre_x']+radius*settings['radius_px']/40*np.cos(angle),settings['centre_y']-radius*settings['radius_px']/40*np.sin(angle)
    colors={'cnn':'#00bce8','median':'#ee9029'}
    for method,boxes in result['boxes'].items():
        for x,y,x1,y1 in boxes:
            angles=np.linspace((x-.5)*2*np.pi/5024,(x1-.5)*2*np.pi/5024,20)
            radii=[40+(v+16-.5-79.5)*.05 for v in (y,y1)]
            th=np.r_[angles,angles[::-1],angles[:1]];rad=np.r_[np.full(20,radii[0]),np.full(20,radii[1]),radii[0]]
            ax.plot(*position(th,rad),color=colors[method],lw=1.5)
    ax.set_title('Imported image' if original is not None else 'Complete weld',fontsize=12,color='#203b50');ax.axis('off')
    score=result['scores'].get('cnn',next(iter(result['scores'].values())))
    row,col=np.unravel_index(np.argmax(score),score.shape)
    columns=(np.arange(64)+col-32)%5024;start=max(0,min(64,row-32))
    zoom=figure.add_subplot(122);zoom.imshow(view[start:start+64][:,columns],origin='lower',cmap='gray',vmin=low,vmax=high)
    for method,boxes in result['boxes'].items():
        for x,y,x1,y1 in boxes:
            middle=(((x+x1)/2-col+2512)%5024-2512)+32
            zoom.add_patch(Rectangle((middle-(x1-x)/2-.5,y-start-.5),x1-x,y1-y,fill=False,edgecolor=colors[method],lw=2))
    zoom.set_xlim(-.5,63.5);zoom.set_ylim(-.5,63.5);zoom.set_xticks([]);zoom.set_yticks([])
    zoom.set_title('Magnified inspection area',fontsize=12,color='#203b50')
    angle=col*2*np.pi/5024;radius=40+(row+16-79.5)*.05
    px,py=position(angle,radius);locator=2.4*(imported['settings']['radius_px']/40 if original is not None else 1)
    ax.add_patch(Circle((px,py),locator,fill=False,ec='#728ca1',lw=1.5,ls='--'))
    figure.add_artist(ConnectionPatch(xyA=(px+locator,py),coordsA=ax.transData,xyB=(-.5,31.5),coordsB=zoom.transData,arrowstyle='->',color='#9bb1c2',lw=1.2))
    legend='   |   '.join(f'{"CNN: cyan" if m=="cnn" else "Median: orange"} ({len(b)} candidate{"s" if len(b)!=1 else ""})' for m,b in result['boxes'].items())
    figure.text(.5,.09,legend,ha='center',fontsize=10,color='#203b50')
    figure.text(.5,.045,'Zoom centred on strongest CNN score; dashed circle marks the displayed area.' if 'cnn' in result['scores'] else 'Zoom centred on strongest Median score; dashed circle marks the displayed area.',ha='center',fontsize=8,color='#61778b')
    if imported:figure.text(.5,.012,'EXPLORATORY EXTERNAL IMAGE | '+imported['intensity_mapping']['kind'],ha='center',fontsize=8,color='#965a13')
    figure.subplots_adjust(left=.025,right=.975,bottom=.18,top=.90,wspace=.13)


def export_result(result, folder=None):
    from matplotlib.figure import Figure
    folder=Path(folder) if folder else OUTPUT/'results'/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True,exist_ok=False)
    arrays={**{m+'_score':v for m,v in result['scores'].items()},**{m+'_mask':v for m,v in result['masks'].items()}}
    np.savez_compressed(folder/'predictions.npz',**arrays)
    record={k:result[k] for k in ('source','source_sha256','identity','boxes','seconds','labels_used')}
    if 'import_metadata' in result:record['import_metadata']=result['import_metadata']
    record.update(physical_validation=False,coordinate_policy='128x5024 scored annulus, half-open periodic boxes; raw rows 16:144',
                  image_contract='Raw 160x5024 TOPO; 40 mm ring radius, 0.05 mm radial pitch; caller-supplied registration',
                  adapter_sha256=study.sha(__file__))
    study.write(folder/'prediction.json',record)
    figure=Figure(figsize=(13,6),dpi=120);draw_result(figure,result);figure.savefig(folder/'inspection.png',dpi=180)
    return folder
