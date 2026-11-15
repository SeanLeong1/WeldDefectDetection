"""Resume completed evaluation stages without mutating frozen scientific code."""
from . import campaign as c,assessment as a,learning as learn
from .assessment import evaluate as evaluate_original


def evaluate():
    if not (c.OUT/'results.json').exists():return evaluate_original()
    frozen=learn.verify_freeze();c.verify_sources()
    if not (c.OUT/'robustness_results.json').exists():a.stress(frozen,learn.model_for(frozen['cnn']))
    if not (c.OUT/'paired_profile/results.json').exists():a.paired_diagnostic(frozen,learn.model_for(frozen['cnn']))
    print('Existing frozen evaluation verified; completed outputs retained',flush=True)
