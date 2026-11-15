from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from box_comparison.core import write, sha, selection_key
from box_comparison.campaign import read
from src.training.train_cnn import WeightedBCEDiceLoss
from .methods import neural_inputs, make_network, predict, select_scores, rectangle_targets


def fit(train, val, folder, arm, seed, multiplier, variant, epochs=20, scale=None, rectangle=False):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    name = f'{arm}_{variant}_{multiplier}x_{seed}' + ('_rectangle' if rectangle else '')
    report, checkpoint = folder/(name+'.json'), folder/(name+'.pt')
    if report.exists():
        result = read(report)
        if sha(checkpoint) != result['checkpoint_sha256']:
            raise ValueError('Completed checkpoint changed.')
        return result
    if checkpoint.exists():
        raise ValueError(f'Unverified partial training preserved: {checkpoint}')
    torch.manual_seed(seed); np.random.seed(seed); torch.use_deterministic_algorithms(True)
    x = neural_inputs(train, variant, scale)
    xv = neural_inputs(val, variant, scale)
    y = train['mask'][:, None].astype(np.float32)
    if rectangle:
        y = np.concatenate([y, rectangle_targets(train['mask'])[:, None]], axis=1)
    positive = train['mask'].reshape(len(x), -1).any(1)
    weights = np.where(positive, .5/max(positive.sum(), 1), .5/max((~positive).sum(), 1))
    sampler = WeightedRandomSampler(torch.from_numpy(weights), len(x), replacement=True,
                                    generator=torch.Generator().manual_seed(seed+1))
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=32, sampler=sampler)
    model = make_network(x.shape[1], rectangle)
    fraction = .5*float(train['mask'][positive].mean())
    loss_fn = WeightedBCEDiceLoss(min(20., float(np.sqrt((1-fraction)/max(fraction, 1e-8)))), dice_weight=.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=.00005)
    augmentation = torch.Generator().manual_seed(seed+2)
    history, best = [], None
    started = time.monotonic()
    for epoch in range(1, epochs+1):
        model.train(); losses = []
        for xb, yb in loader:
            # Apply a shared reflection to every image/reference/variability channel.
            flip = torch.rand(len(xb), generator=augmentation) < .5
            xb, yb = xb.clone(), yb.clone()
            xb[flip] = xb[flip].flip(-1); yb[flip] = yb[flip].flip(-1)
            gain = .9 + .2*torch.rand((len(xb), 1, 1, 1), generator=augmentation)
            offset = -.02 + .04*torch.rand((len(xb), 1, 1, 1), generator=augmentation)
            image_channels = 1 if variant == 'cnn' else 2
            xb[:, :image_channels] = (xb[:, :image_channels]*gain+offset).clamp(0, 1)
            if image_channels == 2:
                xb[:, 2:3] = (xb[:, :1]-xb[:, 1:2]).abs()
            if variant == 'context_scaled':
                # Noise-relative channel is dimensionless; common gain scales signal and noise together.
                pass
            optimizer.zero_grad(set_to_none=True)
            output = model(xb)
            loss = loss_fn(output[:, :1], yb[:, :1])
            if rectangle:
                loss = loss + .25*loss_fn(output[:, 1:2], yb[:, 1:2])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss.')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        entry = dict(epoch=epoch, loss=float(np.mean(losses)))
        if epoch % 5 == 0:
            chosen = select_scores(val, predict(model, xv)); entry.update(chosen)
            if best is None or selection_key(chosen['metrics']) > selection_key(best['metrics']):
                best = dict(epoch=epoch, **chosen)
                torch.save(dict(state_dict=model.state_dict(), channels=x.shape[1], rectangle=rectangle), checkpoint)
            print(f'{name} epoch={epoch}/{epochs} val box F1={chosen["metrics"]["box_f1"]:.4f}', flush=True)
        history.append(entry)
    result = dict(arm=arm, implementation='network', seed=seed, multiplier=multiplier, variant=variant,
                  rectangle=rectangle, scale=scale, **best, history=history,
                  training_targets=len(x), positive_training_targets=int(positive.sum()),
                  training_sequences=len({m['sequence_id'] for m in train['meta']}),
                  epochs=epochs, elapsed_seconds=time.monotonic()-started,
                  checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha(checkpoint))
    write(report, result)
    return result
