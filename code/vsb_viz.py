from matplotlib import pyplot as plt 
import numpy as np 
import feautils
import xgboost as xgb 

settings = feautils.default_settings()

model = np.load(settings['SAVED_MODEL']).ravel()[0]
train_fea = np.load(settings['TRAIN_FEATURES'])

# feature scatter plot
reg = train_fea[0,:]
fault = train_fea[3,:]
X = np.arange(0, reg.shape[0])
plt.scatter(X, reg, color='b', marker='x', label='Normal')
plt.scatter(X, fault, color='r', marker='x', label='Fault')
plt.grid()
plt.legend()
plt.title('LBP feature scatter plot')
plt.show()

## ANALYZE TOP FEATURES
all_fscores = model.get_fscore()
fea_names = list(all_fscores.keys())
print('Total features used by the model = {}'.format(len(fea_names)))

TOP_FEA = 10
FEA_DESC = []
print('Feature descriptions for {} top features'.format(TOP_FEA))
f_vals = list(all_fscores.values())
f_keys = list(all_fscores.keys())
sorted_ind = np.argsort(f_vals)
for i in np.arange(1, TOP_FEA+1):
    fea_ind = int(f_keys[sorted_ind[-1*i]][1:])
    FEA_DESC.append(feautils.fea_desc(fea_ind))

xgb.plot_importance(booster=model, max_num_features=TOP_FEA)
plt.show()