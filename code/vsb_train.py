import pandas as pd
import numpy as np
import feautils
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef
import collections
import random
import time
import argparse
# random seed for repeatability
random.seed(0)
np.random.seed(0)

# get settings
settings=feautils.default_settings()

# set general training params
train_params = {
    'PRED_TH': 0.5,
    'ARTIFICIAL_CLEAN_SAMPS': 4000,
    'ARTIFICIAL_SAMP_STD_FACTOR': 0.1,
    'N_ROUND': 25000, # number of learners
    'COMBINED_PHASE': True # combine all phases
}

# set xgb params
xgb_params = {'nthread': settings['XGB_NTHREAD'],
              'max_depth': 8,
              'eta': 0.001,
              'gamma': 0.1,
              'lambda': 0.1,
              'min_child_weight': 1,
              'objective': 'binary:logistic',
              'subsample': 0.5,
              'colsample_bytree': 0.5,
              'silent': 0
              }

# define the eval function
def xgb_mcc(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result. The metric name must not contain a colon (:) or a space
    # since preds are margin(before logistic transformation, cutoff at 0)
    ypred = (preds > train_params['PRED_TH'])
    return 'xgb_mcc', matthews_corrcoef(labels, ypred)

###########
# Training
###########
def main(args):
    if args.simple:
        # train a simplified classifier
        train_params['N_ROUND'] = 1000
        xgb_params['eta'] = 0.1

    print('-- Training using settings: {}'.format(settings))
    print("-- XGBoost params: {}".format(xgb_params))
    print("-- General training params: {}".format(train_params))

    # load precalculated training features
    allX = np.load(settings['TRAIN_FEATURES'])
    allY = pd.read_csv(settings['TRAINY']).target.values.flatten()

    if train_params['COMBINED_PHASE']:
        allX = allX.reshape(-1, 3 * allX.shape[1])
        allY = np.greater(np.sum(allY.reshape(-1,1).reshape(-1, 3), axis=1), 0).astype(int)

    ####################
    # TRAINING FULL DATA
    allX_expanded, allY_expanded = feautils.equalize_classes(allX, allY, extraSamps=train_params['ARTIFICIAL_CLEAN_SAMPS'], stdNoiseFactor=train_params['ARTIFICIAL_SAMP_STD_FACTOR'])
    print('-- Training: {0}, labels: {1}'.format(allX_expanded.shape, collections.Counter(allY_expanded)))
    dtrain = xgb.DMatrix(allX_expanded, allY_expanded)
    allX_expanded = None
    allY_expanded = None
    # training classifier
    starttime = time.time()
    best_classifier = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=train_params['N_ROUND'], feval=xgb_mcc)
    dtrain = None
    endtime = time.time()
    elapsed_sec = endtime - starttime
    print('-- XGB training completed in {0:.2f} sec ({1:.2f} hrs)'.format(elapsed_sec, elapsed_sec/3600))
    np.save('./best_model.npy', best_classifier)
    np.save(settings['SAVED_MODEL'], best_classifier)
    print('-- Saved the full classifier to {0}'.format(settings['SAVED_MODEL']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSB XGB training")
    parser.add_argument('--simple', action='store_true',
                        help="Simplified model training for fast execution")
    main(parser.parse_args())    