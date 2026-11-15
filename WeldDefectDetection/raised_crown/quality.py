"""Post-freeze image-quality diagnostics; never used to filter or retune scenes."""
import numpy as np
from . import campaign as c,rendering as r

def run():
    norm=c.read(c.OUT/'normalization.json');records=[]
    for split in c.COUNTS:
        rows=c.read(c.OUT/f'datasets/{split}/index.json');clean={}
        for row in rows:
            if row['source_class']=='clean':
                with np.load(row['npz_path']) as a:clean[row['background_id']]=a['quiet'].copy()
        for row in rows:
            with np.load(row['npz_path']) as a:
                raw=a['raw'][0,r.SCORE];quiet=a['quiet'][r.SCORE];mask=a['mask'][r.SCORE]>0
                delta=quiet-clean[row['background_id']][r.SCORE]
                if row['source_class']=='hidden':assert not mask.any() and np.array_equal(delta,np.zeros_like(delta))
                parameters=row['parameters']['defects']
                records.append(dict(scene_id=row['scene_id'],split=split,source_class=row['source_class'],
                    background_id=row['background_id'],low_clip_fraction=float(np.mean(raw<norm['low'])),
                    high_clip_fraction=float(np.mean(raw>norm['high'])),noise_rms=float(np.sqrt(np.mean((raw-quiet)**2))),
                    signal_rms=float(np.sqrt(np.mean(delta**2))),visible_area_px=int(mask.sum()),
                    signal_in_mask=float(np.mean(abs(delta[mask]))) if mask.any() else None,
                    diameter_mm=parameters[0]['params']['diameter_mm'] if parameters else None,
                    radial_offset_mm=parameters[0]['params']['radial_offset_mm'] if parameters else None))
    c.write(c.OUT/'image_quality.json',dict(records=records,selection_effect='None; all scenes retained',
        interpretation='Signal and noise measures in the saved renderer units, not measured SEM detection limits'))
    return records
