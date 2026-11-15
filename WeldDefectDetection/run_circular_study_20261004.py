"""CLI for the versioned circular synthetic thesis study."""
import argparse
import torch
from circular_study import generation as g

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['all','verify','prepare','train_data','val_data','test_data','diagnostics','train','evaluate','report']);a=p.parse_args()
    torch.set_num_threads(4)
    if a.stage=='all':
        from circular_study import diagnostics,training,evaluation,reporting,quiet,pore_convergence,benchmark,tiling_diagnostic,error_diagnosis,quality,native_preview,paired_comparison
        g.prepare()
        for split in ('train','val'):g.generate(split)
        if not (g.OUT/'diagnostics/roi_equivalence.json').exists():diagnostics.run()
        if not (g.OUT/'diagnostics/pore_convergence.json').exists():pore_convergence.run()
        if not (g.OUT/'frozen_models.json').exists():training.run()
        else:evaluation.verify_freeze()
        quiet.generate_test()
        if not (g.OUT/'results.json').exists():evaluation.run()
        else:evaluation.audit()
        if not (g.OUT/'diagnostics/native_circular_previews/verification.json').exists():native_preview.run()
        if not (g.OUT/'runtime_benchmark.json').exists():benchmark.run()
        if not (g.OUT/'diagnostics/tile_origin_sensitivity.json').exists():tiling_diagnostic.run()
        if not (g.OUT/'diagnostics/error_taxonomy.json').exists():error_diagnosis.run()
        if not (g.OUT/'paired_comparisons.json').exists():paired_comparison.run()
        quality.run()
        reporting.run()
    elif a.stage=='verify':
        from circular_study.evaluation import audit
        g.baseline.c.verify_sources(g.OUT);audit()
    elif a.stage=='prepare':g.prepare()
    elif a.stage=='test_data':
        from circular_study.quiet import generate_test
        generate_test()
    elif a.stage.endswith('_data'):g.generate(a.stage.split('_')[0])
    elif a.stage=='diagnostics':
        from circular_study.diagnostics import run
        run()
    elif a.stage=='train':
        from circular_study.training import run
        run()
    elif a.stage=='evaluate':
        from circular_study.evaluation import run
        run()
    elif a.stage=='report':
        from circular_study.reporting import run
        run()
