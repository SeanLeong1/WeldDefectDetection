"""Finish the active run after separately launched development generation.

This is an in-session pipeline driver, not a scheduler. It never starts before
the complete normalization and split indexes are present.
"""
import time
import torch
from . import generation as g,training,evaluation,reporting,quiet


def run():
    torch.set_num_threads(4);deadline=time.monotonic()+7200
    while not (g.OUT/'normalization.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Development generation did not complete')
        time.sleep(10)
    for split,count in (('train',60),('val',20)):
        assert len(g.read(g.OUT/f'datasets/{split}/index.json'))==3*count
    assert g.read(g.OUT/'diagnostics/roi_equivalence.json')['status']=='passed'
    print('Development complete. Starting frozen CV comparison.',flush=True)
    training.run()
    print('Models frozen. Generating fresh confirmation rings.',flush=True)
    quiet.generate_test()
    print('Confirmation generation complete. Evaluating frozen choices.',flush=True)
    evaluation.run()
    reporting.run()
    print('COMPLETE',flush=True)


if __name__=='__main__':run()
