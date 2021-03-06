Below you can find an outline of how to reproduce my solution for the “VSB Power Line Fault Detection” competition. This repo contains source code and documentation for my Gold medal winning solution (4th place):
https://www.kaggle.com/c/vsb-power-line-fault-detection/leaderboard

If you run into any trouble with the setup/code or have any questions please contact me.

# ARCHIVE CONTENTS
* code_package.tgz: origial kaggle model upload (contains original code, a pretrained model, documentation)
* ./data/input/: folder to store raw input data files
* ./data/processed/: folder to store all processed feature files
* ./data/submission/: folder to store submission outputs
* ./data/submission/saved_model_70232.npy: saved pretrained model
* ./data/submission/submission_result_70232.csv: out submission file generated by the above model
* ./code/: all training/prediction code and settings files
* ./code/settings.json: stores all data setup paths and thread configuration for running the code
* ./code/feautils.py: feature extraction script
* ./code/vsb_train.py: model training script
* ./code/vsb_predict.py: model prediction script

# HARDWARE/OS
The following hardware specs were used to create the original solution (no GPUs were used)
* For feature extraction: Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz (24 cores) with 16GB RAM
* For training: Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz (8 cores) with 24GB RAM

All the machines ran
```
Description:    Ubuntu 18.04 LTS
Release:        18.04
Codename:       bionic
```

**The machine running the model training must have 24GB RAM or more. XGB will crash if there is less RAM.**

# SOFTWARE
Python packages are detailed separately in `./requirements.txt`:

    Python 3.7.2

# DATA SETUP
Copy the following files from `vsb-power-line-fault-detection.zip` to `./data/input/`.

* metadata_test.csv
* metadata_train.csv
* test.parquet
* train.parquet
* sample_submission.csv

# ENVIRONMENT SETUP

Skip this step if your environment matches the ones used by the solution. If you are using a different hardware config or data paths, modify `./code/settings.json`.
* FEATURE_NJOBS: number of cores to use to run feature extraction (set to just below the number of physical cores on your machine)
* XGB_NTHREAD: number of threads (nthread) to use for running XGB (set to just below the number of physical cores on your machine)
* TRAINX: raw training parquet file path,
* TRAINY: raw training label file path,
* TESTX: raw test parquet data file
* TEST_META: test meta file
* TRAIN_FEATURES: processed training features file
* TEST_FEATURES: processed testing features file
* TMP_OUTPUT: intermediate outputs (not used currently)
* SUBMISSION: predicted output csv
* SAVED_MODEL: final trained model file

# PREDICTION USING THE PRE-PACKAGED MODEL
A pretrained model is included in the package at:

`./data/submission/saved_model_70232.npy`

If you need to run the model on a new dataset, just replace `./data/input/metadata_test.csv, ./data/input/test.parquet` files by the new dataset files.

Run the following commands starting from the archive folder. All data source and destination paths are specified in `./code/settings.json`. **Note that all the following python invocations will overwrite any existing processed files**.

Rename the existing model

    cp ./data/submission/saved_model_70232.npy ./data/submission/saved_model.npy

Generate test feature set (this could take many hours to run)

    cd code
    python feautils.py --test

Predict on the newly generated featureset

    python vsb_predict.py

# TRAINING A NEW MODEL AND PREDICTING
Run the following commands starting from the archive folder. All data source and destination paths are specified in `./code/settings.json`. **Note that all the python invocations will overwrite any existing processed or model files**.

    cd code

Generate training features (takes approx 6 hours with default settings/data)

    python feautils.py --train

Train XGB model (takes approx 24 hours with default settings/data)

    python vsb_train.py

Generate test features (takes approx 14 hours with default settings/data)

    python feautils.py --test

Predict using the model (takes approx 6.6 minutes with default settings/data)

    python vsb_predict.py

# TRAINING AND TESTING A SIMPLIFIED MODEL
For faster feature extraction and model training with lower accuracy, a simplified feature set and model parameter set can be used. Run the following commands starting from the archive folder. All data source and destination paths are specified in `./code/settings.json`. **Note that all the python invocations will overwrite any existing processed or model files**.

    cd code

Generate training features (takes approx 1.8 hours with default settings/data)

    python feautils.py --train --simple

Train XGB model (takes approx 2.5 minutes with default settings/data)

    python vsb_train.py --simple

Generate test features (takes approx 4.4 hours with default settings/data)

    python feautils.py --test --simple

Predict using the model (takes approx 21 secs with default settings/data)

    python vsb_predict.py
