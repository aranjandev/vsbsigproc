import pandas as pd
import numpy as np
import xgboost as xgb
import collections
import feautils
import vsb_train
import time

# get default paths and settings
settings=feautils.default_settings()

# Prediction
# load pretrained model
print('-- Loading model from: {}'.format(settings['SAVED_MODEL']))
best_classifier = np.load(settings['SAVED_MODEL']).ravel()[0]
# load precalculated features
print('-- Loading test features from: {}'.format(settings['TEST_FEATURES']))
testX = np.load(settings['TEST_FEATURES'])
if vsb_train.train_params['COMBINED_PHASE']:
    testX = testX.reshape(-1, 3 * testX.shape[1])

print('-- Predicting on test data: {0} samples...'.format(testX.shape[0]))
starttime = time.time()
testY = []
for trow in testX:
    dtest = xgb.DMatrix(trow.reshape(1,-1))
    rowY = (best_classifier.predict(dtest, ntree_limit=vsb_train.train_params['N_ROUND']) > vsb_train.train_params['PRED_TH']).astype(int)
    testY.append(rowY.ravel()[0])

testY = np.asarray(testY).reshape(-1,1)
testX = None
dtest = None

if vsb_train.train_params['COMBINED_PHASE']:
    testY = np.repeat(testY, 3) # PHASE COMBINED

endtime = time.time()
print('-- Completed predicting in {0:.2f} sec'.format(endtime-starttime))

submissionY = testY
#print(collections.Counter(submissionY))
meta = pd.read_csv(settings['TEST_META'])
print('-- Creating submission...')
submission_dict = {'signal_id': meta.signal_id.values.flatten(), 'target': submissionY.flatten().astype(int)}
submission_df = pd.DataFrame(submission_dict)
submission_df.to_csv(settings['SUBMISSION'], index=False)
print('-- Saved submission to {0}'.format(settings['SUBMISSION']))