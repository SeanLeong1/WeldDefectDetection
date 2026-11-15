"""Illustrate frozen CNN errors; verify re-rendered counts against saved scores."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_frozen_confirmation import verify_frozen, write
from src.algorithms.classical.cnn_detector import CNNDetector
from src.data.datasimulation_dataset import DataSimulationSegmentationDataset
from src.evaluation.evaluate_dataset import _overlay_image, _predict_mask
from src.utils.file_io import load_config
from src.utils.metrics import evaluate_segmentation_masks


def main(root):
    verify_frozen(root)
    directory = root / 'experiments/static_confirmation_cnn'
    with (directory / 'metrics.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    frozen = json.loads((root / 'frozen_models.json').read_text())
    config = load_config(root / 'jobs/evaluate_static/cnn_evaluation.yaml')
    config['cnn']['checkpoint_path'] = frozen['selected']['cnn']['best_checkpoint']
    detector = CNNDetector(config)
    dataset = DataSimulationSegmentationDataset(
        root / 'datasets/static_confirmation', 'all', image_channel=config['data']['image_channel'],
        mask_key=config['data']['mask_key'], input_size=config['cnn']['input_size'])
    source_classes = {r['sample_id']: r['defect_type'] for r in map(json.loads,
        (root / 'datasets/static_confirmation/samples.jsonl').read_text().splitlines())}
    selected = []
    for cls in sorted(set(source_classes.values())):
        candidates = [r for r in rows if source_classes[r['sample_id']] == cls]
        if cls == 'clean':
            selected.append(max(candidates, key=lambda r: int(r['fp'])))
        else:
            selected.append(min(candidates, key=lambda r: int(r['tp']) /
                max(int(r['tp']) + int(r['fp']) + int(r['fn']), 1)))
    fig, axes = plt.subplots(len(selected), 3, figsize=(10, 2.4 * len(selected)), layout='constrained')
    metadata = []
    for index, row in enumerate(selected):
        sample = dataset[int(row['dataset_index'])]
        if sample['sample_id'] != row['sample_id']:
            raise ValueError('Example dataset index no longer matches the saved evaluation.')
        image = sample['image'][0].detach().cpu().numpy().astype(np.float32)
        target = (sample['mask'][0].detach().cpu().numpy() > .5).astype(np.uint8)
        prediction = _predict_mask('cnn', detector, image)
        counts = evaluate_segmentation_masks(prediction, target)
        if any(int(counts[k]) != int(row[k]) for k in ('tp', 'tn', 'fp', 'fn')):
            raise ValueError('Illustration predictions differ from the frozen evaluation.')
        axes[index, 0].imshow(image, cmap='gray', vmin=0, vmax=1)
        axes[index, 1].imshow(target, cmap='gray', vmin=0, vmax=1)
        axes[index, 2].imshow(_overlay_image(image, prediction, target))
        cls = source_classes[row['sample_id']]
        axes[index, 0].set_title(cls + '\n' + row['sample_id'], fontsize=9)
        axes[index, 1].set_title('Ground-truth mask', fontsize=9)
        axes[index, 2].set_title(f"IoU {counts['iou']:.3f}; FP {row['fp']}; FN {row['fn']}", fontsize=9)
        for ax in axes[index]:
            ax.axis('off')
        metadata.append({'source_class': cls, 'sample_id': row['sample_id'],
                         'confusion_matches_saved_evaluation': True,
                         'predicted_is_defective': row['predicted_is_defective'],
                         'counts': {k: int(counts[k]) for k in ('tp', 'tn', 'fp', 'fn')}})
    fig.suptitle('Selected CNN failures, not a random sample\nGreen: missed truth; red: false positives; yellow: detected truth', fontsize=12)
    fig.savefig(root / 'confirmation_error_examples.png', dpi=140)
    plt.close(fig)
    verify_frozen(root)
    write(root / 'confirmation_error_examples.json', {
        'selection_policy': 'Largest false-positive area among clean images; lowest pixel IoU within each defective source class.',
        'purpose': 'Descriptive error inspection only; no model or threshold selection.',
        'examples': metadata})
    print('Seven class-specific examples match their saved evaluation confusion counts.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root.resolve())
