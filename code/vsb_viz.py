from matplotlib import pyplot as plt 
import numpy as np 
import feautils
import xgboost as xgb 
import importlib

importlib.reload(feautils)

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
f_vals = list(all_fscores.values())
f_keys = list(all_fscores.keys())

# calculating feature details
FEA_DESC = []
DETAIL_FILE = '../docs/feature_details.txt'
print('Storing top {} features to {}'.format(TOP_FEA, DETAIL_FILE))
sorted_ind = np.argsort(f_vals)
for i in np.arange(1, TOP_FEA+1):
    fea_ind = int(f_keys[sorted_ind[-1*i]][1:])
    desc = feautils.feature_detail(fea_ind)
    FEA_DESC.append([fea_ind, desc])

with(open(DETAIL_FILE, 'w')) as file:
    for fea_det in FEA_DESC:    
        file.write('Index {}: Phase={}, Win bound={}, Delta={}, Radius={}, Word={:08b} \n'.format(
            fea_det[0],
            fea_det[1]['phase'],
            fea_det[1]['win_bound'],
            fea_det[1]['delta'],
            fea_det[1]['radius'],
            fea_det[1]['word']
        ))


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
