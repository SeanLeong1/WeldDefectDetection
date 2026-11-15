"""Precompute disjoint stress backgrounds with the unchanged frozen evaluator."""
import argparse
from pathlib import Path
import torch
from . import campaign as c,learning as learn,assessment as assess


def run(indices,worker):
    torch.set_num_threads(4);frozen=learn.verify_freeze();c.verify_sources()
    original_read=c.read;original_write=c.write
    index_path=c.OUT/'datasets/stress/index.json';result_path=c.OUT/'robustness_results.json'
    record=dict(background_indices=indices,worker=worker,source_sha256=c.sha(__file__),
                evaluator_sha256=c.sha(assess.__file__),threads=4,
                policy='Unchanged frozen evaluator; disjoint background scheduling, atomic per-scene receipts. Main process assembles the complete evaluation.')
    original_write(c.OUT/f'execution_logs/stress_assessment_worker_{worker}.json',record)
    def read(path):
        rows=original_read(path)
        return [x for x in rows if x['index'] in indices] if Path(path)==index_path else rows
    def write(path,value):
        path=Path(path)
        if path==result_path:
            return original_write(c.OUT/f'execution_logs/stress_assessment_summary_{worker}.json',value)
        if path.parent==c.OUT/'stress_predictions':
            if path.exists():
                assert original_read(path)==value,'Concurrent evaluation differed'
                return
            temporary=path.with_suffix(f'.worker{worker}.tmp')
            original_write(temporary,value);temporary.replace(path)
        else:original_write(path,value)
    c.read=read;c.write=write
    assess.stress(frozen,learn.model_for(frozen['cnn']))
    print('Disjoint stress worker complete',worker,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,required=True);parser.add_argument('--indices',type=int,nargs='+',required=True)
    args=parser.parse_args();run(args.indices,args.worker)
