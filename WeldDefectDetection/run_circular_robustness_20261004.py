"""Generate and evaluate the frozen circular robustness extension."""
import argparse
from circular_robustness import campaign,native_inputs,assessment_native
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','generate','evaluate','all']);a=p.parse_args()
    if a.stage=='prepare':campaign.prepare()
    if a.stage in ('generate','all'):campaign.generate()
    if a.stage in ('evaluate','all'):
        if not (campaign.OUT/'investigation/localisation_resampling.json').exists():
            from circular_robustness.localisation_check import run
            run()
        native_inputs.generate()
        assessment_native.run()
