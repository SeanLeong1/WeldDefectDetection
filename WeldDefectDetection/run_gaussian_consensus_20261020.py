"""Finite Gaussian-scale experiment, separate from all frozen campaigns.

Run with the repository venv: this_file.py all
Only train/validation data select the scale. Confirmation is generated after freeze.
"""
from pathlib import Path
import argparse
import gc
import random
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'WeldDefectDetection'), str(ROOT / 'DataSimulation')]
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from raised_acquisition import campaign as original, learning, assessment
from raised_crown import learning as base
from circular_study import metrics
from src.algorithms.classical.cnn_detector import TinyDefectSegNet
import run_continuous_median_20261019 as previous
import run_median_thresholds_20261019 as prior

OUT = ROOT / 'DataSimulation/data/output/gaussian_consensus_20261020_v1'
SIGMAS = (0., .5, 1., 2., 4., 8.)
OFFSETS = (-256, -192, -128, -64, 64, 128, 192, 256)
SEED_BASE = 2472000000
TRAIN_SEED = 100606002
COUNT = 24
OP = dict(threshold=12., min_area=15, closing=3)
CNN_OP = dict(threshold=.35, min_area=15, closing=3)
read, write, sha = original.read, original.write, original.sha
cv2.setNumThreads(1)
torch.set_num_threads(4)


def preprocess(raw, sigma):
    # Radial pitch .05 mm; angular pitch measured at the 40-mm centreline.
    if sigma:
        raw = gaussian_filter(raw, (sigma, sigma * .05 / (2*np.pi*40/5024)),
                              mode=('reflect', 'wrap'), truncate=4.)
    return base.normalize(raw, read(original.OUT / 'normalization.json'))


def neighbours(image):
    return np.stack([np.roll(image, -k, axis=1) for k in OFFSETS])


def consensus(context, tolerance):
    """Enumerate all eight one-sample constant-model hypotheses, then refit.

    Most inliers wins; ties use inlier SSE after mean refit, then offset order.
    The target is absent. This is exhaustive sample consensus, not random RANSAC.
    """
    best_count = np.zeros(context.shape[1:], 'i1')
    best_sse = np.full(context.shape[1:], np.inf, 'f4')
    best_mean = np.zeros(context.shape[1:], 'f4')
    for hypothesis in context:
        inside = abs(context - hypothesis) <= tolerance
        count = inside.sum(0)
        mean = (context * inside).sum(0) / count
        sse = (((context - mean)**2) * inside).sum(0)
        take = (count > best_count) | ((count == best_count) & (sse < best_sse))
        best_count[take] = count[take]
        best_sse[take] = sse[take]
        best_mean[take] = mean[take]
    return best_mean


def residuals(image, tolerance):
    context = neighbours(image)
    return dict(median=image - np.median(context, axis=0),
                consensus=image - consensus(context, tolerance))


def robust(samples):
    value = np.concatenate(samples, axis=1)
    centre = np.median(value, axis=1, keepdims=True)
    scale = np.maximum(.003, 1.4826*np.median(abs(value-centre), axis=1, keepdims=True))
    return dict(centre=centre.tolist(), scale=scale.tolist())


def calibrate(clean, sigma):
    images = [preprocess(x, sigma) for x in clean]
    # Fixed train-only tolerance rule. Numerical tolerance adapts to noise scale,
    # while multiplier 3 and all detection/postprocessing settings stay fixed.
    r = robust([(im - np.median(neighbours(im), axis=0))[:, ::4] for im in images])
    tolerance = 3*np.asarray(r['scale'], 'f4')
    values = dict(median=[], consensus=[])
    for im in images:
        for name, value in residuals(im, tolerance).items():
            values[name].append(value[:, ::4])
    return dict(sigma=sigma, tolerance=tolerance.tolist(),
                calibration={k:robust(v) for k,v in values.items()})


def scores(image, spec):
    output = {}
    for name, value in residuals(image, np.asarray(spec['tolerance'], 'f4')).items():
        cal = spec['calibration'][name]
        output[name] = (abs(value-np.asarray(cal['centre'], 'f4')) /
                        np.asarray(cal['scale'], 'f4'))[16:144]
    return output


def check():
    rng = np.random.default_rng(20261020)
    c = rng.normal(size=(8, 3, 17)).astype('f4')
    tol = np.full((3, 1), .4, 'f4')
    fast = consensus(c, tol)
    for y in range(3):
        for x in range(17):
            candidates = []
            for h in c[:, y, x]:
                v = c[:, y, x][abs(c[:, y, x]-h) <= tol[y, 0]]
                candidates.append((len(v), -float(((v-v.mean())**2).sum()), float(v.mean())))
            expected = max(enumerate(candidates), key=lambda p:(p[1][0],p[1][1],-p[0]))[1][2]
            assert abs(fast[y, x]-expected) < 1e-6
    c = np.array([1,1,1,1,1,1,20,30], 'f4')[:,None,None]
    assert consensus(c, np.array([[.1]], 'f4')).item() == 1.
    image = rng.random((160, 5024), dtype=np.float32)
    for sigma in SIGMAS:
        a = preprocess(image, sigma)
        b = preprocess(np.roll(image, 13, 1), sigma)
        assert np.allclose(np.roll(a, 13, 1), b, atol=1e-6)
    # Changing the target alone cannot change its eight reference values.
    im = np.zeros((2, 5024), 'f4'); before = neighbours(im)[:,1,0].copy()
    im[1,0] = 100.
    assert np.array_equal(before, neighbours(im)[:,1,0])
    assert np.array_equal(preprocess(image, 0), base.normalize(image, read(original.OUT/'normalization.json')))
    return dict(status='passed',scalar_consensus_cases=51,minority_outliers_rejected=True,
                angular_wrap_all_scales=True,target_excluded=True,zero_sigma_normalization_parity=True)


def verify_protocol():
    p = read(OUT/'protocol.json')
    for path, digest in p['files'].items():
        assert sha(path) == digest, f'Protocol input changed: {path}'
    return p


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'protocol.json').exists():
        verify_protocol()
        return
    original.verify(); learning.verify_freeze()
    paths = {Path(__file__), Path(previous.__file__), Path(prior.__file__), Path(base.__file__),
             Path(learning.__file__), Path(assessment.__file__), Path(metrics.__file__),
             ROOT/'WeldDefectDetection/src/algorithms/classical/cnn_detector.py',
             original.OUT/'frozen_models.json', original.OUT/'normalization.json'}
    paths.update(original.generation_sources())
    for split in ('train','val'):
        index = original.OUT/f'datasets/{split}/index.json'
        paths.add(index)
        paths.update(Path(m['npz_path']) for m in read(index))
    seen = set()
    for folder in (original.OUT, previous.OUT, prior.OUT):
        for index in list(folder.glob('datasets/*/index.json'))+list(folder.glob('confirmation/index.json')):
            seen.update(m['seed'] for m in read(index))
    assert not seen.intersection(range(SEED_BASE, SEED_BASE+COUNT))
    p = dict(campaign=OUT.name, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        sigma_px=list(SIGMAS), sigma_mm=[s*.05 for s in SIGMAS],
        variance_mm2=[(s*.05)**2 for s in SIGMAS],
        preprocessing='Raw full 160x5024 annulus Gaussian, then frozen train normalization; angular wrap/radial reflect; 4-sigma truncation',
        geometry='Uniform physical scale at 40-mm centreline, approximately isotropic over radial width; not native Cartesian blur',
        neighbour_offsets_px=list(OFFSETS), step_px=1, profile_correction=False, residual_smoothing=False,
        references=['median of eight samples','exhaustive eight one-sample constant hypotheses, majority then inlier SSE, mean refit'],
        consensus_tolerance='3 * per-row clean-training median residual scale; scale=max(.003,1.4826*MAD)',
        detection_calibration='Separate per method/scale, per-row clean-training residual median and MAD, sampled every fourth column',
        fixed_detection=OP, cnn_operating_point=CNN_OP,
        selection='Choose sigma maximizing median strict pore-task validation F1, ties pixel IoU then fewer FP then smaller sigma. Consensus receives same sigma.',
        secondary='Show consensus validation curve and independently best validation sigma, but do not train or compare CNN at that secondary scale',
        primary_population='clean + open_keyhole + open_intrinsic; box IoU >= .5',
        cnn=dict(architecture='TinyDefectSegNet, in_channels=1, base_channels=24', arm='hard_negative',
                 initialization='from scratch', seed=TRAIN_SEED, epochs=20, patches_per_epoch=1200,
                 recipe='Existing AdamW/cosine/BCE+Dice/clean hard-negative mining, unchanged; one prespecified seed, no seed or threshold search'),
        confirmation=dict(seed_base=SEED_BASE, backgrounds=COUNT, scenes=96,
                          coordinates='declared nominal', generate='Only after freeze'),
        constraints='No guaranteed CNN superiority. Never tune on confirmation. Old baseline retained unchanged.',
        files={str(p):sha(p) for p in paths}, physical_validation=False)
    write(OUT/'self_checks.json', check())
    write(OUT/'protocol.json', p)
    print('PROTOCOL AND CHECKS PASSED', flush=True)


def develop():
    verify_protocol()
    if (OUT/'selection.json').exists():
        return
    assert not (OUT/'confirmation').exists()
    clean = []
    for meta in read(original.OUT/'datasets/train/index.json'):
        if meta['source_class']=='clean':
            with np.load(meta['npz_path']) as z:
                assert not z['mask'].any()
                clean.append(z['raw'][0])
    val = read(original.OUT/'datasets/val/index.json')
    trials = {}
    for sigma in SIGMAS:
        key = str(sigma)
        folder = OUT/f'validation/sigma_{key}'; folder.mkdir(parents=True,exist_ok=True)
        cp = folder/'calibration.json'
        if not cp.exists(): write(cp, calibrate(clean, sigma))
        spec = read(cp); rows = dict(median=[],consensus=[])
        for i,meta in enumerate(val):
            rp = folder/(meta['scene_id']+'.json')
            if rp.exists(): result = read(rp)
            else:
                with np.load(meta['npz_path']) as z:
                    image=preprocess(z['raw'][0],sigma);mask=z['mask'][16:144]
                result={k:prior.rows_for(s,mask,meta,OP)['0.50'] for k,s in scores(image,spec).items()}
                write(rp,result)
            for k in rows: rows[k].append(result[k])
            if i%16==15: print('VALIDATION',sigma,i+1,flush=True)
        trials[key]={k:prior.summary(v) for k,v in rows.items()}
        write(folder/'summary.json',trials[key])
        print('SCALE',sigma,trials[key],flush=True)
    selected=max(SIGMAS,key=lambda s:(*metrics.key(trials[str(s)]['median']['primary_pores']),-s))
    secondary=max(SIGMAS,key=lambda s:(*metrics.key(trials[str(s)]['consensus']['primary_pores']),-s))
    write(OUT/'selection.json',dict(selected_sigma=selected,consensus_secondary_sigma=secondary,trials=trials,
          protocol_sha256=sha(OUT/'protocol.json')))
    print('SELECTED MEDIAN SCALE',selected,flush=True)


def load_smoothed(split,sigma):
    data=learning.load(split)
    for i,meta in enumerate(data['meta']):
        with np.load(meta['npz_path']) as z:data['x'][i,0]=preprocess(z['raw'][0],sigma)
    return data


def train():
    verify_protocol()
    sigma=read(OUT/'selection.json')['selected_sigma']
    if (OUT/'cnn.json').exists():
        spec=read(OUT/'cnn.json');assert sha(spec['checkpoint'])==spec['checkpoint_sha256'];return
    assert not (OUT/'confirmation').exists()
    data=load_smoothed('train',sigma)
    positive,negative=base.populations(data)
    px=np.mean([base.tile(data['fullmask'][a],b,d).mean() for a,b,d in positive])*.5
    weight=float(min(20,np.sqrt((1-px)/max(px,1e-8))))
    random.seed(TRAIN_SEED);np.random.seed(TRAIN_SEED);torch.manual_seed(TRAIN_SEED)
    torch.use_deterministic_algorithms(True)
    model=TinyDefectSegNet(in_channels=1,base_channels=24)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,20,eta_min=5e-5)
    candidates=learning.clean_candidates(data,TRAIN_SEED)
    checkpoint=OUT/'cnn.pt';history=[];mining=[];hard=None;start=0;started=time.monotonic()
    if checkpoint.exists():
        s=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(s['state_dict']);optimizer.load_state_dict(s['optimizer'])
        scheduler.load_state_dict(s['scheduler']);torch.set_rng_state(s['torch_rng'])
        history=s['history'];mining=s['mining'];hard=s['hard'];start=s['epoch']
        assert s['sigma']==sigma and s['protocol_sha256']==sha(OUT/'protocol.json')
    for epoch in range(start,20):
        if epoch in (5,10,15):
            hard,record=learning.mine(data,model,candidates);mining.append(dict(after_epoch=epoch,**record))
        ids=learning.epoch_indices(positive,negative,hard,TRAIN_SEED,epoch)
        model.train();losses=[]
        for j in range(0,len(ids),32):
            x,y=base.batch(data,ids[j:j+32]);target=torch.from_numpy(y)
            optimizer.zero_grad(set_to_none=True);logits=model(torch.from_numpy(x));prob=torch.sigmoid(logits)
            loss=F.binary_cross_entropy_with_logits(logits,target,pos_weight=torch.tensor(weight))+1-(2*(prob*target).sum()+1)/(prob.sum()+target.sum()+1)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(loss.item())
        scheduler.step();history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
        temp=checkpoint.with_suffix('.tmp')
        torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            torch_rng=torch.get_rng_state(),history=history,mining=mining,hard=hard,epoch=epoch+1,
            sigma=sigma,protocol_sha256=sha(OUT/'protocol.json')),temp);temp.replace(checkpoint)
        print('CNN EPOCH',epoch+1,round(np.mean(losses),4),flush=True)
    del data;gc.collect();model.eval()
    val=load_smoothed('val',sigma)
    rows=metrics.evaluate(val,base.predict(val,model),CNN_OP)
    write(OUT/'cnn.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),seed=TRAIN_SEED,
          sigma=sigma,parameters=sum(p.numel() for p in model.parameters()),history=history,mining=mining,
          selection=dict(best=CNN_OP),validation=prior.summary(rows),seconds=time.monotonic()-started))
    print('CNN TRAINED',flush=True)


def freeze():
    verify_protocol()
    if (OUT/'frozen.json').exists():verify();return
    assert not (OUT/'confirmation').exists()
    chosen=read(OUT/'selection.json')
    files={p:sha(p) for p in read(OUT/'protocol.json')['files']}
    files.update({str(p):sha(p) for p in OUT.rglob('*.json')})
    files[str(OUT/'cnn.pt')]=sha(OUT/'cnn.pt')
    sources=[Path(p) for p in files if Path(p).suffix=='.py']
    with zipfile.ZipFile(OUT/'sources_at_freeze.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,p.relative_to(ROOT))
    write(OUT/'frozen.json',dict(selected_sigma=chosen['selected_sigma'],
        calibration=read(OUT/f'validation/sigma_{chosen["selected_sigma"]}/calibration.json'),
        cnn=read(OUT/'cnn.json'),files=files,
        frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print('FROZEN BEFORE CONFIRMATION',flush=True)


def verify():
    f=read(OUT/'frozen.json')
    for p,h in f['files'].items():assert sha(p)==h,f'Frozen input changed: {p}'
    return f


def background(index):
    previous.OUT=OUT;previous.BASE_SEED=SEED_BASE;previous.COUNT=COUNT;previous.verify_frozen=verify
    previous.background(index)


def generate(workers):
    import subprocess
    verify();folder=OUT/'confirmation';folder.mkdir(exist_ok=True)
    logs=OUT/'logs';logs.mkdir(exist_ok=True);pending=list(range(COUNT));active=[]
    while pending or active:
        while pending and len(active)<workers:
            i=pending.pop(0);receipt=folder/f'background_{i:03d}.json'
            if receipt.exists():
                for row in read(receipt):assert sha(row['npz_path'])==row['npz_sha256']
                continue
            handle=(logs/f'background_{i:03d}.log').open('a',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,str(Path(__file__)),'background','--index',str(i)],cwd=ROOT,
                stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active.append((proc,handle,i))
        for proc,handle,i in active[:]:
            if proc.poll() is not None:
                handle.close();active.remove((proc,handle,i))
                if proc.returncode:raise RuntimeError(f'Background {i} failed; see logs')
                print('BACKGROUND COMPLETE',i,flush=True)
        if active:time.sleep(2)
    write(folder/'index.json',[r for i in range(COUNT) for r in read(folder/f'background_{i:03d}.json')])


def evaluate():
    f=verify();old=learning.verify_freeze()
    models={'cnn_same_scale':base.model_for(f['cnn']),'original_cnn_unfiltered':base.model_for(old['cnn'])}
    index=read(OUT/'confirmation/index.json');folder=OUT/'predictions';folder.mkdir(exist_ok=True)
    allrows={}
    for meta in index:
        assert sha(meta['npz_path'])==meta['npz_sha256']
        assert meta['freeze_sha256']==sha(OUT/'frozen.json')
        rp=folder/(meta['scene_id']+'.json')
        if rp.exists():
            receipt=read(rp);assert receipt['freeze_sha256']==sha(OUT/'frozen.json')
        else:
            with np.load(meta['npz_path']) as z:raw=z['raw'][0];mask=z['mask'][16:144]
            image=preprocess(raw,f['selected_sigma'])
            s=scores(image,f['calibration'])
            s['cnn_same_scale']=base.predict_image(image,models['cnn_same_scale'])
            unsmoothed=preprocess(raw,0)
            s['original_median_unfiltered']=base.median_image(unsmoothed,old['median']['spec'])
            s['original_cnn_unfiltered']=base.predict_image(unsmoothed,models['original_cnn_unfiltered'])
            rows={}
            for name,score in s.items():
                op=CNN_OP if name=='cnn_same_scale' else old['cnn']['selection']['best'] if name=='original_cnn_unfiltered' else OP
                rows[name]=prior.rows_for(score,mask,meta,op,True)['0.50']
            receipt=dict(rows=rows,freeze_sha256=sha(OUT/'frozen.json'),source_sha256=meta['npz_sha256'])
            write(rp,receipt)
            # Prespecified first background, no selection for favourable images.
            if meta['index']==0:np.savez_compressed(folder/(meta['scene_id']+'_scores.npz'),**s)
        assert receipt['source_sha256']==meta['npz_sha256']
        for name,row in receipt['rows'].items():allrows.setdefault(name,[]).append(row)
        print('SCORED',meta['scene_id'],flush=True)
    results={name:dict(**prior.summary(rows,True),rows=rows) for name,rows in allrows.items()}
    differences={name:dict(primary_pores=assessment.paired_difference(previous.subset(allrows['cnn_same_scale']),previous.subset(allrows[name])),
                          all_conditions=assessment.paired_difference(allrows['cnn_same_scale'],allrows[name]))
                 for name in ('median','consensus')}
    write(OUT/'results.json',dict(selected_sigma=f['selected_sigma'],results=results,
          cnn_minus_classical=differences,freeze_sha256=sha(OUT/'frozen.json'),physical_validation=False))
    verify();assert len(index)==96 and len({r['background_id'] for r in index})==24
    assert {r['seed'] for r in index}==set(range(SEED_BASE,SEED_BASE+COUNT))
    for name,rows in allrows.items():
        assert results[name]['primary_pores']['tp']+results[name]['primary_pores']['fn']==48
    write(OUT/'audit.json',dict(status='passed',scenes=96,backgrounds=24,
        frozen_inputs_unchanged=True,selection_preceded_confirmation=True,physical_validation=False))
    print('FINAL', {k:v['primary_pores'] for k,v in results.items()},flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['all','prepare','develop','train','freeze','generate','background','evaluate'])
    p.add_argument('--workers',type=int,default=3);p.add_argument('--index',type=int)
    args=p.parse_args()
    for name,fn in [('prepare',prepare),('develop',develop),('train',train),('freeze',freeze),
                    ('generate',lambda:generate(args.workers)),('evaluate',evaluate)]:
        if args.stage in ('all',name):fn()
    if args.stage=='background':background(args.index)


if __name__=='__main__':main()
