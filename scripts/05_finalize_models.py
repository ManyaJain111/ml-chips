"""Step 05: select the best model per case, evaluate on TEST, save (Deliverable 5).  Owner: P2.

Outputs: artifacts/models/{case}.joblib            (the 4 saved model files)
         results/tables/chosen_models_test.{csv,md} (test median + p95 error per case/target)
         results/plots/pred_vs_spice_{case}.{png,pdf}
"""

from __future__ import annotations

import argparse

from leakml.config import DEFAULT_CONFIG


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--quick", action="store_true", help="pilot run with small sample counts")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)  # noqa: F841  (remove the noqa once main() is implemented)
    # TODO(P2):
    # 1. load the candidates + the validation metrics from step 04
    # 2. best = ml.train.select_best(candidates, cfg.ml.selection_metric), and record WHY
    # 3. ml.train.evaluate_on_test(best, splits["test"]) -> table + pred-vs-SPICE plot per case
    # 4. ml.train.save_model(best, artifacts/models/{case}.joblib)
    # 5. smoke test: TransistorModelBank.load_all(...).predict on 5 test rows gives the same numbers
    raise NotImplementedError


if __name__ == "__main__":
    main()
