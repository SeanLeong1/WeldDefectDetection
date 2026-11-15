"""Bounded validation-only selection for the four box-producing approaches."""
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.classical.cnn_detector import TinyDefectSegNet
from src.training.train_cnn import WeightedBCEDiceLoss
from .core import (model_input, morphology_score, select_threshold, selection_key,
                   sha, write)


def inputs(data, hybrid):
    return np.stack([model_input(t, r, hybrid) for t, r in zip(data['target'], data['reference'])])


def predict_network(model, x):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([torch.sigmoid(model(torch.from_numpy(x[i:i+32]))).numpy()[:, 0]
                               for i in range(0, len(x), 32)])


def fit_network(train, val, root, arm, seed, epochs=10):
    root = Path(root)
    receipt = root / f'{arm}_{seed}.json'
    checkpoint = root / f'{arm}_{seed}.pt'
    if receipt.exists():
        import json
        saved = json.loads(receipt.read_text())
        if sha(checkpoint) != saved['checkpoint_sha256']:
            raise ValueError('Completed training checkpoint changed.')
        return saved
    if checkpoint.exists():
        raise ValueError(f'Partial training exists at {checkpoint}; preserve it and use a new run directory.')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    hybrid = arm == 'cnn_median'
    xtrain, xval = inputs(train, hybrid), inputs(val, hybrid)
    ytrain = train['mask'][:, None].astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(xtrain), torch.from_numpy(ytrain)),
                        batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed))
    model = TinyDefectSegNet(in_channels=3 if hybrid else 1)
    positive_weight = float(min(20, np.sqrt((ytrain.size-ytrain.sum())/max(ytrain.sum(), 1))))
    loss_fn = WeightedBCEDiceLoss(positive_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    history, best = [], None
    started = time.monotonic()
    for epoch in range(1, epochs+1):
        model.train(); losses = []
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        selected, _ = select_threshold(val, predict_network(model, xval), [.2, .35, .5, .65, .8])
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), **selected))
        if best is None or selection_key(selected['metrics']) > selection_key(best['metrics']):
            best = dict(epoch=epoch, **selected)
            root.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state_dict=model.state_dict(), arm=arm, seed=seed,
                            epoch=epoch, in_channels=3 if hybrid else 1), checkpoint)
        print(f'{arm} seed={seed} epoch={epoch}/{epochs} val_box_f1={selected["metrics"]["box_f1"]:.4f}', flush=True)
    result = dict(arm=arm, seed=seed, epochs=epochs, selected=best, history=history,
                  positive_weight=positive_weight, parameter_count=sum(p.numel() for p in model.parameters()),
                  checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha(checkpoint),
                  elapsed_seconds=time.monotonic()-started)
    write(receipt, result)
    return result


def fit_simple(val, root):
    best, candidates = None, []
    for polarity in ('dark', 'bright', 'both'):
        for kernel in (9, 17, 31):
            scores = np.stack([morphology_score(t, kernel, polarity) for t in val['target']])
            selected, trials = select_threshold(val, scores, [.02, .04, .06, .08, .12, .18], openings=(1, 3))
            for trial in trials:
                trial['parameters'].update(kernel_size=kernel, polarity=polarity)
            candidates.extend(trials)
            if best is None or selection_key(selected['metrics']) > selection_key(best['metrics']):
                best = selected
    write(Path(root)/'thresholding_selection.json', dict(selected=best, candidates=candidates))
    median, trials = select_threshold(val, np.abs(val['target']-val['reference']),
                                     [.02, .04, .06, .08, .12, .18], openings=(1, 3))
    write(Path(root)/'median_selection.json', dict(selected=median, candidates=trials))
    return dict(thresholding=best, median=median)


def scores_for(data, specification):
    arm = specification['arm']
    if arm == 'thresholding':
        p = specification['parameters']
        return np.stack([morphology_score(t, p['kernel_size'], p['polarity']) for t in data['target']])
    if arm == 'median':
        return np.abs(data['target']-data['reference'])
    checkpoint = Path(specification['checkpoint'])
    if sha(checkpoint) != specification['checkpoint_sha256']:
        raise ValueError('Frozen model hash mismatch.')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = TinyDefectSegNet(in_channels=saved['in_channels'])
    model.load_state_dict(saved['state_dict'])
    return predict_network(model, inputs(data, arm == 'cnn_median'))
