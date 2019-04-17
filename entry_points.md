# change to code folder
cd code

# Open settings.json and make sure "FEATURE_NJOBS" and "XGB_NTHREAD" are set
# correctly based on your system.

# Run feature extraction for train and test (takes many hours to complete)
# This script reads raw parquet input file
# Preprocess the signals, extracts features, saves them
# Input and output paths are defined in: code/settings.json
python feautils.py --train --test

# Run XGBoost training (takes many hours to complete)
# It reads extracted features in the previous step
# Runs XGBoost on the features and saves the classifier
# Input and output paths are defined in: code/settings.json
python vsb_train.py

# Runs prediction on a given set of pre-extracted features
# Saves the output for "SUBMISSION"
python vsb_predict.py
