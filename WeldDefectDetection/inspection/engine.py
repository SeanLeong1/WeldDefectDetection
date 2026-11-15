"""Validated imports, tiled frozen inference, and auditable per-file exports.

Display imports are exploratory. They are never represented as raw calibrated TOPO.
No file is resized to imitate the training physical scale.
"""
from __future__ import annotations

import csv
import hashlib
import json
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageOps

REPO = Path(__file__).resolve().parents[2]
LEGACY_CAMPAIGN = REPO / 'DataSimulation/data/output/variability_campaign_20260926_v1'
DEFAULT_CAMPAIGN = REPO / 'DataSimulation/data/output/circular_study_20261004_v4'
EXTENSIONS = {'.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp', '.npy', '.npz'}
MAX_PIXELS = 16_000_000
MAX_BYTES = 256 * 1024 * 1024
ARMS = ('cnn', 'median', 'thresholding')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


@dataclass(frozen=True)
class Options:
    mode: str = 'display'
    pixel_size_mm: float | None = None
    convert_colour: bool = False
    recursive: bool = False
    arms: tuple[str, ...] = ('cnn',)

    def check(self):
        if self.mode not in ('raw', 'normalized', 'display'):
            raise ValueError('Choose raw, normalized or display input.')
        if not self.arms or any(a not in ARMS for a in self.arms):
            raise ValueError('Choose at least one supported CV method.')
        if self.pixel_size_mm is not None and (not np.isfinite(self.pixel_size_mm) or self.pixel_size_mm <= 0):
            raise ValueError('Pixel size must be finite and positive, or blank if unknown.')


def discover(source, recursive=False):
    source = Path(source).resolve()
    if not source.exists():
        raise ValueError('Input path does not exist.')
    if source.is_file():
        if source.suffix.lower() not in EXTENSIONS:
            raise ValueError('Unsupported file type. Use PNG, TIFF, JPEG, BMP, NPY or NPZ.')
        return [source]
    files = sorted((p for p in (source.rglob('*') if recursive else source.iterdir())
                    if p.is_file() and not p.is_symlink() and p.suffix.lower() in EXTENSIONS),
                   key=lambda p: str(p).casefold())
    if not files:
        raise ValueError('No supported images found in this folder.')
    if len(files) > 2000:
        raise ValueError('More than 2,000 files found. Select a smaller folder.')
    return files


def _array_ok(value, name):
    a = np.asarray(value)
    if a.dtype.kind not in 'uif' or not np.isfinite(a).all():
        raise ValueError(f'{name} must contain finite numeric values, without objects.')
    return a


def load_image(path, options: Options, norm):
    if norm.get('_layout')=='circular_annulus':
        from .circular import load_annulus
        return load_annulus(path,options,norm)
    options.check()
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('File exceeds the 256 MiB import limit.')
    warnings = []
    spacing = options.pixel_size_mm
    metadata = {}
    sidecar = path.with_name(path.name+'.json')
    if sidecar.exists():
        if sidecar.stat().st_size > 65536:
            raise ValueError('Image metadata sidecar exceeds 64 KiB.')
        metadata = json.loads(sidecar.read_text(encoding='utf-8-sig'))
        if not isinstance(metadata, dict):
            raise ValueError('Metadata sidecar must be a JSON object.')
        if metadata.get('pixel_size_mm') is not None:
            declared = float(metadata['pixel_size_mm'])
            if spacing is not None and not np.isclose(spacing, declared, rtol=1e-6, atol=1e-9):
                raise ValueError('Entered pixel size disagrees with image sidecar metadata.')
            spacing = declared
        if options.mode != 'display' and metadata.get('channel', 'topo_magnitude') != 'topo_magnitude':
            raise ValueError('Raw/normalized input requires the matching TOPO magnitude channel, not a signed difference or other modality.')
        if metadata.get('source_kind') == 'synthetic_preview':
            warnings.append('Generator preview: this input is synthetic and is not the frozen training distribution.')
    context = None
    encoded_max = None
    if path.suffix.lower() == '.npz':
        with zipfile.ZipFile(path) as archive:
            if sum(i.file_size for i in archive.infolist()) > MAX_BYTES:
                raise ValueError('Expanded NPZ exceeds the 256 MiB import limit.')
        with np.load(path, allow_pickle=False) as archive:
            key = next((k for k in ('target_raw', 'raw', 'topo') if k in archive), None)
            if key is None:
                raise ValueError('NPZ needs target_raw, raw or topo. Labels are never used as input.')
            array = archive[key].copy()
            if 'context_raw' in archive:
                context = archive['context_raw'].copy()
            if 'pixel_size_mm' in archive:
                embedded = float(np.asarray(archive['pixel_size_mm']).item())
                if spacing is not None and not np.isclose(spacing, embedded, rtol=1e-6, atol=1e-9):
                    raise ValueError('Entered pixel size disagrees with NPZ metadata.')
                spacing = embedded
        if options.mode != 'raw':
            raise ValueError('Raw NPZ input requires the Raw TOPO input mode.')
    elif path.suffix.lower() == '.npy':
        array = np.load(path, allow_pickle=False, mmap_mode='r')
    else:
        with Image.open(path) as im:
            if getattr(im, 'n_frames', 1) != 1:
                raise ValueError('Multi-page/animated images are ambiguous. Export individual frames first.')
            if im.width * im.height > MAX_PIXELS:
                raise ValueError('Image exceeds 16 megapixels. Select a smaller region.')
            if im.mode == 'P':
                im = im.convert('RGB')
            if 'A' in im.getbands():
                raise ValueError('Transparent images need an explicit opaque grayscale export.')
            im = ImageOps.exif_transpose(im)
            array = np.asarray(im).copy()
            encoded_max = 65535. if im.mode.startswith('I;16') else (255. if im.mode in ('L', 'RGB') else None)
    array = _array_ok(array, 'Image')
    if array.ndim == 3 and array.shape[-1] == 3:
        if np.array_equal(array[..., 0], array[..., 1]) and np.array_equal(array[..., 0], array[..., 2]):
            array = array[..., 0]
        elif options.convert_colour and options.mode == 'display':
            array = array[..., 0] * .299 + array[..., 1] * .587 + array[..., 2] * .114
            warnings.append('Colour converted to luminance; this does not establish TOPO compatibility.')
        else:
            raise ValueError('Colour image: export grayscale or explicitly enable exploratory colour conversion.')
    if array.ndim != 2 or min(array.shape) < 8 or array.size > MAX_PIXELS:
        raise ValueError('Expected one grayscale image, at least 8 pixels per side and at most 16 megapixels.')
    if float(array.max()) == float(array.min()):
        raise ValueError('Image has no intensity variation. Check the channel or export.')
    if spacing is not None and (not np.isfinite(spacing) or spacing <= 0):
        raise ValueError('Pixel-size metadata must be finite and positive.')
    if options.mode in ('raw', 'normalized'):
        if spacing is None or not np.isclose(spacing, .05, rtol=0, atol=1e-9):
            raise ValueError('Frozen raw/normalized inference requires 0.05 mm/pixel; images are not rescaled automatically.')
    if options.mode == 'display':
        warnings.append('Exploratory display-image transfer: acquisition and intensity mapping are unvalidated.')
        if spacing is None:
            warnings.append('Physical scale unknown; no measurements in mm are exported.')
        elif not np.isclose(spacing, .05, rtol=0, atol=1e-9):
            warnings.append('Physical scale differs from training; no resampling was applied.')
        if encoded_max is None:
            if array.dtype == np.uint8:
                encoded_max = 255.
            elif array.dtype == np.uint16:
                encoded_max = 65535.
            else:
                raise ValueError('Display mode needs 8/16-bit integer pixels. Choose Raw or Normalized for float arrays.')
        target = array.astype(np.float32) / encoded_max
    elif options.mode == 'normalized':
        if array.min() < 0 or array.max() > 1:
            raise ValueError('Normalized input must already be in [0,1], using the frozen training mapping.')
        target = array.astype(np.float32)
        warnings.append('Caller declares that the frozen training normalization has already been applied.')
    else:
        def normalize(a):
            return np.sqrt(np.clip((a.astype(np.float32)-norm['raw_low']) /
                                  (norm['raw_high']-norm['raw_low']), 0, 1)).astype(np.float32)
        target = normalize(array)
        clipped = float(np.mean((array < norm['raw_low']) | (array > norm['raw_high'])))
        if clipped > .01:
            warnings.append(f'{clipped:.1%} of raw pixels clip under frozen normalization.')
        if context is not None:
            context = _array_ok(context, 'Context')
            if context.shape != (4, 64, 64) or array.shape != (64, 64):
                raise ValueError('Median NPZ requires a 64x64 target and four 64x64 neighbouring crops.')
            context = normalize(context)
    if target.shape != (64, 64):
        warnings.append('Full-image tiling/stitching is a prototype extension beyond the frozen crop evaluation.')
    return dict(target=target, context=context, pixel_size_mm=spacing, warnings=warnings,
                original_dtype=str(array.dtype), shape=list(array.shape), mode=options.mode,
                metadata_sha256=digest(sidecar) if sidecar.exists() else None)


class FrozenInspector:
    def __new__(cls,campaign=DEFAULT_CAMPAIGN):
        record=json.loads((Path(campaign)/'frozen_models.json').read_text())
        if 'cnn_candidates' in record:
            from .circular import CircularInspector
            return CircularInspector(campaign)
        return super().__new__(cls)

    def __init__(self, campaign=DEFAULT_CAMPAIGN):
        self.campaign = Path(campaign).resolve()
        self.frozen = json.loads((self.campaign/'frozen_models.json').read_text())
        if digest(self.campaign/'normalization.json') != self.frozen['normalization_sha256']:
            raise ValueError('Frozen normalization hash mismatch.')
        self.norm = json.loads((self.campaign/'normalization.json').read_text())
        self.models = {}
        self.model_hashes = {}

    def _network(self, arm):
        if arm not in self.models:
            import torch
            from box_development.methods import make_network
            spec = self.frozen['models'][arm]
            # Locate only within the selected campaign; never follow a stale absolute path.
            filename = Path(spec['checkpoint'].replace('\\', '/')).name
            path = self.campaign/'training'/filename
            if digest(path) != spec['checkpoint_sha256']:
                raise ValueError('Frozen checkpoint hash mismatch.')
            model = make_network(1)
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['state_dict'])
            model.eval()
            self.models[arm] = model
            self.model_hashes[arm] = digest(path)
        return self.models[arm]

    def score(self, item, arm, cancel=None):
        from box_development.methods import simple_scores
        spec = self.frozen['models'][arm]
        target = item['target']
        if arm == 'median':
            if item['context'] is None:
                raise ValueError('Median requires a raw NPZ with four actual spatial neighbours. An arbitrary photo/folder provides no validated context.')
            data = dict(target=target[None], context=item['context'][None])
            return simple_scores(data, spec)[0]
        height, width = target.shape
        padded = np.pad(target, ((0, max(0, 64-height)), (0, max(0, 64-width))), mode='edge')
        def starts(n):
            return sorted(set(list(range(0, n-63, 64)) + [n-64]))
        positions = [(y, x) for y in starts(padded.shape[0]) for x in starts(padded.shape[1])]
        sums = np.zeros_like(padded)
        counts = np.zeros_like(padded)
        for offset in range(0, len(positions), 32):
            if cancel is not None and cancel.is_set():
                raise InterruptedError('Cancelled before next tile batch.')
            batch = positions[offset:offset+32]
            tiles = np.stack([padded[y:y+64, x:x+64] for y, x in batch])
            if arm == 'cnn':
                import torch
                with torch.inference_mode():
                    scores = torch.sigmoid(self._network(arm)(torch.from_numpy(tiles[:, None]))).numpy()[:, 0]
            else:
                scores = simple_scores(dict(target=tiles, context=np.zeros((len(tiles), 4, 64, 64), np.float32)), spec)
            for (y, x), score in zip(batch, scores):
                sums[y:y+64, x:x+64] += score
                counts[y:y+64, x:x+64] += 1
        return (sums/counts)[:height, :width]

    def predict(self, item, arm, cancel=None):
        from box_refinement.methods import predicted_regions
        score = self.score(item, arm, cancel)
        parameters = self.frozen['models'][arm]['parameters']
        mask, boxes = predicted_regions(score, **parameters)
        return score, mask, boxes


def run(source, output_root, options=Options(), *, campaign=DEFAULT_CAMPAIGN,
        validate_only=False, progress: Callable | None = None, cancel: threading.Event | None = None):
    options.check()
    files = discover(source, options.recursive)
    inspector = FrozenInspector(campaign)
    output_root = Path(output_root).resolve()
    source_path = Path(source).resolve()
    if source_path.is_dir() and (output_root == source_path or source_path in output_root.parents):
        raise ValueError('Choose an output folder outside the input folder to avoid re-importing results.')
    out = output_root/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    records = []
    seen = set()
    if not validate_only:
        import torch
        torch.set_num_threads(4)
    for index, path in enumerate(files):
        if cancel is not None and cancel.is_set():
            break
        base = dict(source=str(path), index=index+1)
        try:
            if path.stat().st_size > MAX_BYTES:
                raise ValueError('File exceeds the 256 MiB import limit.')
            file_hash = digest(path)
            base['input_sha256'] = file_hash
            if file_hash in seen:
                records.append(dict(**base, method='', status='duplicate', message='Identical file already processed.'))
                continue
            seen.add(file_hash)
            item = load_image(path, options, inspector.norm)
            for arm in options.arms:
                try:
                    if arm == 'median' and item['context'] is None:
                        raise ValueError('Median needs NPZ context_raw containing four actual spatial neighbours.')
                    if arm == 'cnn':
                        inspector._network(arm)  # Validate checkpoint even for preflight.
                    record = dict(**base, method=arm, status='validated' if validate_only else 'complete',
                                  shape=item['shape'], pixel_size_mm=item['pixel_size_mm'],
                                  metadata_sha256=item['metadata_sha256'], original_dtype=item['original_dtype'],
                                  warnings=item['warnings'], mode=options.mode, physical_validation=False,
                                  parameters=inspector.frozen['models'][arm]['parameters'])
                    if not validate_only:
                        start = time.perf_counter()
                        score, mask, boxes = inspector.predict(item, arm, cancel)
                        record.update(boxes_xyxy=boxes, candidates=len(boxes), seconds=time.perf_counter()-start,
                                      coordinate_policy='half-open original-image pixel indices')
                        spacing = item['pixel_size_mm']
                        record['boxes_mm'] = [[v*spacing for v in b] for b in boxes] if spacing is not None else None
                        if hasattr(inspector,'coordinates'):record.update(inspector.coordinates(boxes))
                        prefix = f'{index+1:04d}_{arm}'
                        np.savez_compressed(out/f'{prefix}_prediction.npz', score=score, mask=mask)
                        overlay = Image.fromarray(np.uint8(np.clip(item['target'], 0, 1)*255)).convert('RGB')
                        drawing = ImageDraw.Draw(overlay)
                        for x0, y0, x1, y1 in boxes:
                            drawing.rectangle((x0, y0, x1-1, y1-1), outline='#19e4b1', width=max(1, round(overlay.width/400)))
                            if hasattr(inspector,'coordinates') and x1>overlay.width:
                                drawing.rectangle((0,y0,x1-overlay.width-1,y1-1),outline='#19e4b1',width=2)
                        overlay.save(out/f'{prefix}_overlay.png')
                        record['overlay'] = str(out/f'{prefix}_overlay.png')
                        if hasattr(inspector,'coordinates'):
                            centre=(boxes[0][0]+boxes[0][2])//2 if boxes else 0
                            columns=(np.arange(128)+centre-64)%overlay.width
                            zoom=Image.fromarray(np.asarray(overlay)[:,columns]).resize((768,384),Image.Resampling.NEAREST)
                            zoom.save(out/f'{prefix}_preview.png');record['preview']=str(out/f'{prefix}_preview.png')
                            record['warnings']=list(record['warnings'])+['Preview shows the first candidate neighbourhood, or the 0-degree neighbourhood when none. Complete ring is saved in overlay.']
                        record['prediction'] = str(out/f'{prefix}_prediction.npz')
                    records.append(record)
                except InterruptedError:
                    records.append(dict(**base, method=arm, status='cancelled', message='Cancelled.'))
                    break
                except Exception as exc:
                    records.append(dict(**base, method=arm, status='error', message=str(exc)))
        except Exception as exc:
            records.append(dict(**base, method='', status='error', message=str(exc)))
        finally:
            if progress:
                progress(index+1, len(files), records[-1] if records else base)
    summary = dict(schema='weld-inspection/v1', campaign=str(inspector.campaign),
                   engine_sha256=digest(__file__), created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   frozen_models_sha256=digest(inspector.campaign/'frozen_models.json'),
                   normalization_sha256=digest(inspector.campaign/'normalization.json'),
                   checkpoint_sha256=inspector.model_hashes,
                   source=str(source_path), output=str(out), options=options.__dict__,
                   files_discovered=len(files), cancelled=bool(cancel and cancel.is_set()),
                   physical_validation=False, records=records)
    if hasattr(inspector,'source_hashes'):
        summary['inference_source_sha256']=inspector.source_hashes()
    write_json(out/'results.json', summary)
    with (out/'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        fields = ['source', 'method', 'status', 'candidates', 'seconds', 'message', 'warnings']
        writer = csv.DictWriter(f, fields, extrasaction='ignore')
        writer.writeheader()
        for r in records:
            row = {**r, 'warnings': '; '.join(r.get('warnings', []))}
            # Prevent spreadsheet formula interpretation of untrusted filenames/errors.
            writer.writerow({k: "'"+v if isinstance(v, str) and v.startswith(('=', '+', '-', '@')) else v for k, v in row.items()})
    return summary
