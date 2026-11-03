"""Part A transistor ML models.  Owner: P2.

split.py      reproducible train/val/test split
features.py   feature engineering + target transform (log|I| and sign handling)
zoo.py        the 4 model types being compared
train.py      fit, compare on validation, select, test-set evaluation, save
predictor.py  TransistorModel / TransistorModelBank: the ONLY API P3/P4 call (SPEC §4.6)
"""
