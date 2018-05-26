import numpy as np
import pandas as pd


class ItemKNN:
    '''
    ItemKNN(n_sims = 100, lmbd = 20, alpha = 0.5, session_key = 'SessionId', item_key = 'ItemId', time_key = 'Time')

    Item-to-item predictor that computes the the similarity to all items to the given item.

    Similarity of two items is given by:

    .. math::
        s_{i,j}=\sum_{s}I\{(s,i)\in D & (s,j)\in D\} / (supp_i+\\lambda)^{\\alpha}(supp_j+\\lambda)^{1-\\alpha}

    Parameters
    --------
    n_sims : int
        Only give back non-zero scores to the N most similar items. Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
    lmbd : float
        Regularization. Discounts the similarity of rare items (incidental co-occurrences). (Default value: 20)
    alpha : float
        Balance between normalizing with the supports of the two items. 0.5 gives cosine similarity, 1.0 gives confidence (as in association rules).
    session_key : string
        header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        header of the timestamp column in the input file (default: 'Time')

    '''

    def __init__(self, n_sims=100, lmbd=20, alpha=0.5, session_key='SessionId', item_key='ItemId', time_key='Time'):
        self.n_sims = n_sims
        self.lmbd = lmbd
        self.alpha = alpha
        self.item_key = item_key
        self.session_key = session_key
        self.time_key = time_key

    def fit(self, data):
        '''
        Trains the predictor.

        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).

        '''
        data.set_index(np.arange(len(data)), inplace=True)
        itemids = data[self.item_key].unique()
        n_items = len(itemids)
        print("So luong item: ", n_items)
        data = pd.merge(data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': np.arange(len(itemids))}),
                        on=self.item_key, how='inner')
        sessionids = data[self.session_key].unique()
        print("So luong Session: ", len(sessionids))
        data = pd.merge(data, pd.DataFrame({self.session_key: sessionids, 'SessionIdx': np.arange(len(sessionids))}),
                        on=self.session_key, how='inner')
        supp = data.groupby('SessionIdx').size() #so hang cua session hien tai
        session_offsets = np.zeros(len(supp) + 1, dtype=np.int32)
        session_offsets[1:] = supp.cumsum() #session tu hang bao nhieu den hang bao nhieu (khi nhom theo SessionIdx)
        index_by_sessions = data.sort_values(['SessionIdx', self.time_key]).index.values #sap xep theo SessionIdx, cot 2 chi ra cac dong xuat hien SessionIdx do
        supp = data.groupby('ItemIdx').size() #so hang co chua item co nhan ItemIdx
        item_offsets = np.zeros(n_items + 1, dtype=np.int32)
        item_offsets[1:] = supp.cumsum() #item tu hang nao den hang nao (khi nhom theo ItemIdx)
        index_by_items = data.sort_values(['ItemIdx', self.time_key]).index.values #sắp xếp theo ItemIdx, cột 2 chỉ ra các dòng xuất hiện ItemIdx do, sau do su dung offset se co duoc danh sach cac dong chua item
        self.sims = dict()
        for i in range(n_items):
            if i%100 == 0:
                print(i)
            iarray = np.zeros(n_items)
            start = item_offsets[i] #dong dau tien xuat hien itemIdx
            end = item_offsets[i + 1] #dong cuoi xuat hien itemIdx
            for e in index_by_items[start:end]: #e la index cua dong
                uidx = data.SessionIdx.values[e] #sessionIdx cua dong e
                ustart = session_offsets[uidx] #dong bat dau sessionIdx
                uend = session_offsets[uidx + 1] #dong ket thuc sessionIdx
                user_events = index_by_sessions[ustart:uend] #cac dong co trong sessionIdx
                iarray[data.ItemIdx.values[user_events]] += 1 #data.ItemIdx.values[user_events]: ItemIdx cua cac item trong session
            iarray[i] = 0 #vecto [N_items,1]
            norm = np.power((supp[i] + self.lmbd), self.alpha) * np.power((supp.values + self.lmbd), (1.0 - self.alpha))
            norm[norm == 0] = 1
            iarray = iarray / norm
            print(iarray)
            indices = np.argsort(iarray)[-1:-1 - self.n_sims:-1] #lay n_sim phan tu co gia tri cao nhat (lay gia tri ItemIdx)
            self.sims[itemids[i]] = pd.Series(data=iarray[indices], index=itemids[indices]) #luu lai 1 bang data: diem tuong dong voi item trong indices, index: itemid trong tap du lieu

    def predict_next(self, session_id, input_item_id, predict_for_item_ids):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.

        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.

        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.

        '''
        preds = np.zeros(len(predict_for_item_ids))
        sim_list = self.sims[input_item_id]
        mask = np.in1d(predict_for_item_ids, sim_list.index) #phan tu nao cua sim_list nam trong predict_for_items
        preds[mask] = sim_list[predict_for_item_ids[mask]]
        return pd.Series(data=preds, index=predict_for_item_ids)