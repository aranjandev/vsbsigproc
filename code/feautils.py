import argparse
from numba import jit
import pandas as pd
import pyarrow.parquet as pq
from matplotlib import pyplot as plt
import time
import numpy as np
import collections
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier
import time
import datetime
import h5py
import scipy.signal
from joblib import Parallel, delayed
import json

PHASE = 3
WORDLEN = 8
RADIICOUNT = 6
RADII = np.multiply(np.power(2,np.arange(0,RADIICOUNT)), WORDLEN)
DELTA = np.asarray([0,4,8])
ALL_WIN_BOUNDS = np.asarray([[0.0,1.0], [0.0,0.25], [0.25,0.5], [0.5,0.75], [0.75,1.0]])

def default_settings(setting_file='./settings.json'):
    jd = json.JSONDecoder()
    settings = jd.decode(open(setting_file).read())
    return settings

def default_feaparam():
    feaparam = {'medfilt_win': 101}
    return feaparam

# from feature index to feature generation description
def fea_desc(f_index):
    # word, radius, delta, win_bound, phase
    TOTAL_WORDS = 2**WORDLEN
    phase = int(np.floor(f_index/(ALL_WIN_BOUNDS.shape[0] * TOTAL_WORDS * RADIICOUNT * DELTA.size)))
    leftovers = f_index - (phase * ALL_WIN_BOUNDS.shape[0] * TOTAL_WORDS * RADIICOUNT * DELTA.size)
    win_bound = int(np.floor(leftovers/(TOTAL_WORDS * RADIICOUNT * DELTA.size)))
    leftovers = leftovers - (win_bound * TOTAL_WORDS * RADIICOUNT * DELTA.size) 
    delta = int(np.floor(leftovers/(TOTAL_WORDS * RADIICOUNT)))
    leftovers = leftovers - (delta * TOTAL_WORDS * RADIICOUNT)
    radius = int(np.floor(leftovers/TOTAL_WORDS))
    leftovers = leftovers - (radius * TOTAL_WORDS)
    word = int(leftovers)
    print('Index {}: Phase={}, Win bound={}, Delta={}, Radius={}, Word={:08b}'.format(
        f_index,
        phase,
        ALL_WIN_BOUNDS[win_bound],
        DELTA[delta],
        RADII[radius],
        word    
    ))
    return {'phase': phase, 'win_bound': win_bound, 'delta': delta, 'radius': radius, 'word': word}

# pyramidal lbp computation with pyramid of windows
# radii must be multiple of word
# pass a column vector
# for fast optimization: remove all params other than vector arg.
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def pyrlbp_multiscale(inputsig):
    wordlen = WORDLEN
    radiicount = RADIICOUNT
    radii = RADII #np.multiply(np.power(2,np.arange(0,radiicount)), wordlen)
    delta = DELTA # [0,4,8]
    all_win_bounds = ALL_WIN_BOUNDS #ALL_WIN_BOUNDS #[[0.0,1.0], [0.0,0.25], [0.25,0.5], [0.5,0.75], [0.75,1.0]]
    powers = np.zeros((wordlen,1))
    samp_count = int(wordlen/2)
    pyrlbp_all = np.zeros((0), dtype=np.float64)
    for i in range(wordlen):
        powers[i] = 2 ** i
    for w_ind in np.arange(0, all_win_bounds.shape[0]):
        win_bound = all_win_bounds[w_ind,:]
        lbp_mult_delta = np.zeros((0), dtype=np.float64)
        win_start = round(win_bound[0] * float(inputsig.shape[0]))
        win_end = round(win_bound[1] * float(inputsig.shape[0]))
        if win_start < 0 or win_end > inputsig.shape[0] or win_end <= win_start:
            continue;
        vector = inputsig[win_start:win_end]
        for d in delta:
            lbp_ms = np.zeros((0), dtype=np.float64)
            for r in radii:
                lbp = np.zeros((2**wordlen), dtype=np.float64)
                factor = int(r/samp_count)
                for i in range(r, vector.shape[0]-r):
                    # center element to compare
                    x = vector[i]
                    # calculate before i
                    full_buff = vector[i-r:i]
                    # select word elements from this buffer
                    sel_buff = full_buff[0::factor]
                    bin_rep_before = np.greater_equal(sel_buff, x + d).astype(np.float64).reshape(1,-1)
                    # calculate after i
                    full_buff = vector[i+1:i+r+1]
                    # select word elements from this buffer
                    sel_buff = full_buff[0::factor]
                    bin_rep_after = np.greater_equal(sel_buff, x).astype(np.float64).reshape(1,-1)
                    # convert to decimal number
                    full_bin_rep = np.hstack((bin_rep_before, bin_rep_after))
                    int_word = full_bin_rep.dot(powers).ravel()[0]
                    hist_index = int(int_word)
                    lbp[hist_index] = lbp[hist_index] + 1.0
                lbp_ms = np.hstack((lbp_ms, lbp))
            lbp_mult_delta = np.hstack((lbp_mult_delta, lbp_ms))
        pyrlbp_all = np.hstack((pyrlbp_all, lbp_mult_delta))
    return pyrlbp_all

# Simplified fast feature
# radii must be multiple of word
# pass a column vector
# for fast optimization: remove all params other than vector arg.
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def pyrlbp_multiscale_simple(inputsig):
    wordlen = WORDLEN
    radiicount = RADIICOUNT
    radii = RADII #np.multiply(np.power(2,np.arange(0,radiicount)), wordlen)
    delta = [DELTA[0]]
    all_win_bounds = ALL_WIN_BOUNDS[0,:].reshape(1,-1)
    powers = np.zeros((wordlen,1))
    samp_count = int(wordlen/2)
    pyrlbp_all = np.zeros((0), dtype=np.float64)
    for i in range(wordlen):
        powers[i] = 2 ** i    
    for w_ind in np.arange(0, all_win_bounds.shape[0]):
        win_bound = all_win_bounds[w_ind,:]
        lbp_mult_delta = np.zeros((0), dtype=np.float64)
        win_start = round(win_bound[0] * float(inputsig.shape[0]))
        win_end = round(win_bound[1] * float(inputsig.shape[0]))
        if win_start < 0 or win_end > inputsig.shape[0] or win_end <= win_start:
            continue;
        vector = inputsig[win_start:win_end]
        for d in delta:
            lbp_ms = np.zeros((0), dtype=np.float64)
            for r in radii:
                lbp = np.zeros((2**wordlen), dtype=np.float64)
                factor = int(r/samp_count)
                for i in range(r, vector.shape[0]-r):
                    # center element to compare
                    x = vector[i]
                    # calculate before i
                    full_buff = vector[i-r:i]
                    # select word elements from this buffer
                    sel_buff = full_buff[0::factor]
                    bin_rep_before = np.greater_equal(sel_buff, x + d).astype(np.float64).reshape(1,-1)
                    # calculate after i
                    full_buff = vector[i+1:i+r+1]
                    # select word elements from this buffer
                    sel_buff = full_buff[0::factor]
                    bin_rep_after = np.greater_equal(sel_buff, x).astype(np.float64).reshape(1,-1)
                    # convert to decimal number
                    full_bin_rep = np.hstack((bin_rep_before, bin_rep_after))
                    int_word = full_bin_rep.dot(powers).ravel()[0]
                    hist_index = int(int_word)
                    lbp[hist_index] = lbp[hist_index] + 1.0
                lbp_ms = np.hstack((lbp_ms, lbp))
            lbp_mult_delta = np.hstack((lbp_mult_delta, lbp_ms))
        pyrlbp_all = np.hstack((pyrlbp_all, lbp_mult_delta))
    return pyrlbp_all

# apply proc_fun on the median filtered filename col data
def process_medfilt_filecol(proc_fun, filename, col):
    feaParams = default_feaparam()
    x = pq.read_pandas(filename, columns=[col]).to_pandas().values.reshape(-1,1).astype(np.float64)
    filtsig = np.subtract(x, scipy.signal.medfilt(x, [feaParams['medfilt_win'],1]))
    return proc_fun(filtsig)

# extract feature using function: medfilt_lbp_filecol
def parallel_feature_parquet_cols(parquet_file, meta_file, fea_proc_fun, columns=[], bSave=False, savepath=[], n_jobs=8):
    starttime = time.time()
    meta = pd.read_csv(meta_file)
    if len(columns) == 0:
        columns = list(range(0,len(meta.signal_id)))
    col_ids = meta.signal_id[columns]

    print('-- Parallel processing data matrix: {0}'.format(col_ids))
    st = time.time()
    allProcessedOut = Parallel(n_jobs=n_jobs, verbose=8)(delayed(process_medfilt_filecol)(proc_fun=fea_proc_fun, filename=parquet_file, col=str(ii)) for ii in col_ids)
    outMat = np.asarray(allProcessedOut)
    et = time.time()
    print('- Total time calculating features {0} = {1:.2f} sec'.format(outMat.shape, et-st))
    if bSave:
        if len(savepath) == 0:
            savepath = './tmp.npy'
        print('- Saving all test features of shape {0} to: {1}'.format(outMat.shape, savepath))
        np.save(savepath, outMat)
    return outMat

# generates increment new rows in inmat by adding a small noise to random samples
# noise added to a column <= noise_factor * std(column)
def expand_samples_std_noise(inmat, new_count, max_noise_factor):
    max_noise = np.tile(np.multiply(np.std(inmat, axis=0).reshape(1,-1), max_noise_factor), [new_count, 1])
    multipliers = np.random.rand(new_count, inmat.shape[1])
    if new_count > inmat.shape[0]:
        replace = True
    else:
        replace = False
    sel_ind = np.random.choice(inmat.shape[0], new_count, replace=replace).ravel()
    extra_rows = inmat[sel_ind,:] + np.multiply(multipliers, max_noise)
    return np.vstack((inmat, extra_rows))

# makes number of samples per class equal by artificially generating adequate samples
def equalize_classes(trainX, trainY, extraSamps=0, stdNoiseFactor=0.01):
    pd_idx = np.argwhere(np.equal(trainY, 1)).ravel()
    clean_idx = np.argwhere(np.equal(trainY, 0)).ravel()
    new_count_pd = clean_idx.shape[0] - pd_idx.shape[0] + extraSamps
    new_count_clean = extraSamps
    trainX_expanded = np.vstack((expand_samples_std_noise(trainX[pd_idx,:], new_count_pd, stdNoiseFactor), expand_samples_std_noise(trainX[clean_idx,:], new_count_clean, stdNoiseFactor)))
    trainY_expanded = np.vstack((np.ones((pd_idx.shape[0]+new_count_pd,1)), np.zeros((clean_idx.shape[0]+new_count_clean, 1)))).ravel()
    return trainX_expanded, trainY_expanded

def main(args):
    settings = default_settings()
    feaParams = default_feaparam()
    if args.simple:
        fea_proc_fun = pyrlbp_multiscale_simple
    else:
        fea_proc_fun = pyrlbp_multiscale
    print('-- Processing feature using: {0}'.format(fea_proc_fun))
    if args.train:
        parallel_feature_parquet_cols(settings['TRAINX'], settings['TRAINY'], fea_proc_fun=fea_proc_fun, bSave=True, savepath=settings['TRAIN_FEATURES'], n_jobs=settings['FEATURE_NJOBS'])
    if args.test:
        parallel_feature_parquet_cols(settings['TESTX'], settings['TEST_META'], fea_proc_fun=fea_proc_fun, bSave=True, savepath=settings['TEST_FEATURES'], n_jobs=settings['FEATURE_NJOBS'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSB feature extraction")
    parser.add_argument('--train', action='store_true',
                        help="Extract training features")
    parser.add_argument('--test', action='store_true',
                        help="Extract test features")
    parser.add_argument('--simple', action='store_true',
                        help="Simplified feature extraction for fast execution")
    main(parser.parse_args())
