from matplotlib import pyplot as plt 
import numpy as np 
import feautils
import xgboost as xgb 

bShow = False
settings = feautils.default_settings()

model = np.load(settings['SAVED_MODEL']).ravel()[0]
train_fea = np.load(settings['TRAIN_FEATURES'])

# feature scatter plot
reg = train_fea[0,:]
fault = train_fea[3,:]
X = np.arange(0, reg.shape[0])
plt.figure()
plt.scatter(X, reg, color='b', marker='x', label='Normal')
plt.scatter(X, fault, color='r', marker='x', label='Fault')
plt.grid()
plt.legend()
plt.title('LBP feature scatter plot')
plt.savefig(fname='../docs/lbp_scatter.svg', format='svg')
if bShow:
    plt.show()

## ANALYZE TOP FEATURES
all_fscores = model.get_fscore()
fea_names = list(all_fscores.keys())
print('Total features used by the model = {}'.format(len(fea_names)))

TOP_FEA = 20
FEA_DESC = []
print('Feature descriptions for {} top features'.format(TOP_FEA))
f_vals = list(all_fscores.values())
f_keys = list(all_fscores.keys())
sorted_ind = np.argsort(f_vals)
for i in np.arange(1, TOP_FEA+1):
    fea_ind = int(f_keys[sorted_ind[-1*i]][1:])
    FEA_DESC.append(feautils.fea_desc(fea_ind))

plt.figure()
xgb.plot_importance(booster=model, max_num_features=TOP_FEA)
plt.savefig(fname='../docs/fea_importance.svg', format='svg')
if bShow:
    plt.show()

# Show feature weight distribution
wt = np.asarray(f_vals)
wt_norm = np.divide(wt, np.max(wt))
log_wts = np.log10(wt_norm) 
plt.figure()
plt.hist(log_wts, bins=100)
plt.grid()
plt.title('Histogram of log10 of weights in range [{:.2f}, {:.2f}]'.format(np.min(log_wts), np.max(log_wts)))
plt.savefig(fname='../docs/wt_hist.svg', format='svg')
if bShow:
    plt.show()
