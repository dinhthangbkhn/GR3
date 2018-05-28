import numpy as np
import pandas as pd

class ItemKNN():
    def __init__(self, n_sims = 200, alpha = 0.5):
        self.n_sims = n_sims
        self.alpha = alpha
        self.item_key = 'ItemId'
        self.session_key= 'SessionId'
        self.time_key = 'Time'

    def fit(self, data):
        data.set_index(np.arange(len(data)), inplace=True)

        itemids = data[self.item_key].unique()
        n_items = len(itemids)
        data = pd.merge(data, pd.DataFrame({'ItemId': itemids, 'ItemIdx': np.arange(n_items)}),on='ItemId', how='inner' )

        sessionids = data[self.session_key].unique()
        n_sessions = len(sessionids)
        data = pd.merge(data, pd.DataFrame({'SessionId': sessionids, 'SessionIdx': np.arange(n_sessions)}),
                        on='SessionId', how='inner')

        supp = data.groupby('SessionIdx').size()
        session_offset = np.zeros(n_sessions+1, dtype=np.int32)
        session_offset[1:] = supp.cumsum()
        index_by_sessions = data.sort_values(['SessionIdx', self.time_key]).index.values

        supp = data.groupby('ItemIdx').size()
        item_offset = np.zeros(n_items + 1, dtype=np.int32)
        item_offset[1:] = supp.cumsum()
        index_by_items = data.sort_values(['ItemIdx', 'Time']).index.values
        print(" ")

        self.sims = dict()
        for i in range(n_items):
            sim_array = np.zeros(n_items)
            start_line_item = item_offset[i]
            end_line_item = item_offset[i+1]
            for line in index_by_items[start_line_item: end_line_item]:
                sess_id = data['SessionIdx'].values[line]
                start_line_sess = session_offset[sess_id]
                end_line_sess = session_offset[sess_id+1]
                line_sess = index_by_sessions[start_line_sess:end_line_sess]
                items_of_sess = data.ItemIdx.values[line_sess]
                sim_array[items_of_sess]+=1
            sim_array[i] = 0
            norm = np.power(supp[i]+20, self.alpha) * np.power(supp.values+20, 1-self.alpha)
            norm[norm == 0] = 1
            sim_array = sim_array/norm
            indices = np.argsort(sim_array)[-1:-1 - self.n_sims:-1]
            self.sims[itemids[i]] = pd.Series(data=sim_array[indices], index=itemids[indices])

    def predict_next(self,sessionid, input_item_id, predict_items):
        preds = np.zeros(len(predict_items))
        sim_list = self.sims[input_item_id]
        mask = np.in1d(predict_items, sim_list.index)  # phan tu nao cua sim_list nam trong predict_for_items
        preds[mask] = sim_list[predict_items[mask]]
        return pd.Series(data=preds, index=predict_items)

def evaluate_ItemKNN(pr, test_data, train_data, items=None, cut_off=20, session_key='SessionId', item_key='ItemId',
                      time_key='Time'):
    '''
    Evaluates the baselines wrt. recommendation accuracy measured by recall@N and MRR@N.
    '''
    test_data.sort_values([session_key, time_key], inplace=True)
    items_to_predict = train_data[item_key].unique()
    evalutation_point_count = 0
    prev_iid, prev_sid = -1, -1
    mrr, recall = 0.0, 0.0
    for i in range(len(test_data)):
        sid = test_data[session_key].values[i]
        iid = test_data[item_key].values[i]
        if prev_sid != sid:
            prev_sid = sid
        else:
            if items is not None:
                if np.in1d(iid, items):
                    items_to_predict = items
                else:
                    items_to_predict = np.hstack(([iid], items))
            preds = pr.predict_next(sid, prev_iid, items_to_predict)
            preds[np.isnan(preds)] = 0
            preds += 1e-8 * np.random.rand(len(preds))  # Breaking up ties

            rank = (preds > preds[iid]).sum() + 1

            if i%500 == 0:
                print(i)
            assert rank > 0
            if rank < cut_off:
                recall += 1
                mrr += 1.0 / rank
            evalutation_point_count += 1
        prev_iid = iid
    return recall / evalutation_point_count, mrr / evalutation_point_count