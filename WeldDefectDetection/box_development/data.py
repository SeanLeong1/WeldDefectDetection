from pathlib import Path
import numpy as np
from box_comparison.core import load_windows, subset
from src.data.temporal_window_dataset import TemporalWindowDataset


def load(root, namespace=None, training_only=False):
    data = load_windows(root)
    ds = TemporalWindowDataset(root, 'all')
    data['context'] = np.stack([ds[i]['context'].numpy() for i in range(len(ds))])
    prefix = namespace or Path(root).name
    for meta in data['meta']:
        meta['dataset_root'] = str(Path(root).resolve())
        meta['original_split'] = meta['split']
        for key in ('window_id', 'sequence_id'):
            meta[key] = prefix + ':' + meta[key]
        if training_only:
            meta['split'] = 'train'
    return data


def concatenate(parts):
    result = {k: sum([p[k] for p in parts], []) if k == 'meta'
              else np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if len({m['window_id'] for m in result['meta']}) != len(result['meta']):
        raise ValueError('Duplicate training window identity.')
    if any(m['split'] != 'train' for m in result['meta']):
        raise ValueError('Training assembly contains a held-out window.')
    return result


def training_sets(root, dev):
    base = subset(load(dev, 'original'), 'train')
    batches = [load(root/'datasets'/f'train_{i}', f'addition_{i}', True) for i in range(1, 4)]
    return {1: base, 2: concatenate([base, batches[0]]), 4: concatenate([base, *batches])}


def shuffle_context(data):
    groups = {}
    for i, meta in enumerate(data['meta']):
        groups.setdefault(meta['sequence_id'], []).append(i)
    ids = list(groups)
    if len(ids) < 2:
        raise ValueError('Context shuffling requires multiple sequences.')
    np.random.default_rng(921770099).shuffle(ids)
    context = data['context'].copy()
    for i, key in enumerate(ids):
        dest, source = groups[key], groups[ids[(i+1) % len(ids)]]
        if len(dest) != len(source):
            raise ValueError('Unequal sequence lengths.')
        context[dest] = data['context'][source]
    return {**data, 'context': context, 'reference': np.median(context, axis=1)}
