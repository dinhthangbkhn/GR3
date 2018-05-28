# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def evaluate_sessions_batch(model, train_data, test_data, cut_off=20, batch_size=100, session_key='SessionId',
                            item_key='ItemId', time_key='Time'):

    model.predict = False
    itemids = train_data[item_key].unique()
    itemidmap = pd.Series(data=np.arange(len(itemids)), index=itemids)

    test_data.sort_values([session_key, time_key], ascending=True)
    offset_sessions = np.zeros(test_data[session_key].nunique() + 1, dtype=np.int32)
    offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
    evalutation_point_count = 0
    mrr, recall = 0.0, 0.0
    if len(offset_sessions) - 1 < batch_size:
        batch_size = len(offset_sessions) - 1
    iters = np.arange(batch_size).astype(np.int32)
    maxiter = iters.max()
    start = offset_sessions[iters]
    end = offset_sessions[iters + 1]
    in_idx = np.zeros(batch_size, dtype=np.int32)
    np.random.seed(42)
    while True:
        valid_mask = iters >= 0
        if valid_mask.sum() == 0:
            break
        start_valid = start[valid_mask]
        minlen = (end[valid_mask] - start_valid).min()
        in_idx[valid_mask] = test_data[item_key].values[start_valid]
        for i in range(minlen - 1):
            out_idx = test_data[item_key].values[start_valid + i + 1]
            preds = model.predict_next_batch(iters, in_idx, itemidmap, batch_size)
            preds.fillna(0, inplace=True)
            in_idx[valid_mask] = out_idx
            ranks = (preds.values.T[valid_mask].T > np.diag(preds.ix[in_idx].values)[valid_mask]).sum(axis=0) + 1
            rank_ok = ranks < cut_off
            recall += rank_ok.sum()
            mrr += (1.0 / ranks[rank_ok]).sum()
            evalutation_point_count += len(ranks)
        start = start + minlen - 1
        mask = np.arange(len(iters))[(valid_mask) & (end - start <= 1)]
        for idx in mask:
            maxiter += 1
            if maxiter >= len(offset_sessions) - 1:
                iters[idx] = -1
            else:
                iters[idx] = maxiter
                start[idx] = offset_sessions[maxiter]
                end[idx] = offset_sessions[maxiter + 1]
    return recall / evalutation_point_count, mrr / evalutation_point_count


