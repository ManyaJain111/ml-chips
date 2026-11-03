"""Step 04: split, train the 4 model types per case, compare on validation (Deliverable 4).  Owner: P2.

Outputs: artifacts/splits/{case}.npz
         results/tables/model_comparison.{csv,md}
         artifacts/candidates/{case}_{model}.joblib   (all fitted candidates, so step 05 needn't refit)
"""

from __future__ import annotations

import argparse

from leakml.config import DEFAULT_CONFIG


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--quick", action="store_true", help="pilot run with small sample counts")
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--models", nargs="*", default=None, help="subset of zoo.MODEL_TYPES")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)  # noqa: F841  (remove the noqa once main() is implemented)
    # TODO(P2):
    # 1. for each case: df = load_dataset(case); splits = ml.split.split_dataset(df, scheme, seeds.split)
    #    save_split_indices -> artifacts/splits/{case}.npz
    # 2. results[case] = ml.train.compare_models(case, splits, cfg.ml.model_types, seeds.training)
    # 3. ml.train.comparison_table(results) -> results/tables/model_comparison.{csv,md}
    # 4. persist every candidate (joblib) so step 05 can pick one without refitting
    # NEVER look at splits["test"] in this script.
    raise NotImplementedError


if __name__ == "__main__":
    main()
