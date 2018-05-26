import tensorflow as tf
import pandas as pd
import numpy as np


class GRU4Rec:
    def __init__(self, sess, args):
        self.sess = sess
        self.is_training = args.is_training

        self.layers = args.layers
        self.rnn_size = args.rnn_size
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.dropout_p_hidden = args.dropout_p_hidden
        self.learning_rate = args.learning_rate
        self.session_key = args.session_key
        self.item_key = args.item_key
        self.time_key = args.time_key
        self.n_items = args.n_items
        self.weight_decay = args.weight_decay
        self.optimize = args.optimize

        if args.hidden_act == 'tanh':
            self.hidden_act = self.tanh
        elif args.hidden_act == 'relu':
            self.hidden_act = self.relu
        else:
            raise NotImplementedError

        if args.loss == 'cross-entropy':
            if args.final_act == 'tanh':
                self.final_activation = self.softmaxth
            else:
                self.final_activation = self.softmax
            self.loss_function = self.cross_entropy
        elif args.loss == 'bpr':
            if args.final_act == 'linear':
                self.final_activation = self.linear
            elif args.final_act == 'relu':
                self.final_activation = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.bpr
        elif args.loss == 'top1':
            if args.final_act == 'linear':
                self.final_activation = self.linear
            elif args.final_act == 'relu':
                self.final_activatin = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.top1
        elif args.loss == 'top1_max_wd':
            if args.final_act == 'linear':
                self.final_activation = self.linear
            elif args.final_act == 'relu':
                self.final_activatin = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.top1_max_decay
        elif args.loss == 'bpr_max':
            if args.final_act == 'linear':
                self.final_activation = self.linear
            elif args.final_act == 'relu':
                self.final_activation = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.bpr_max
        else:
            raise NotImplementedError
        self.checkpoint_dir = args.checkpoint_dir

        self.build_model()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        if self.is_training:
            return
        self.predict_state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in range(self.layers)]
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, '{}/gru-model-{}'.format(self.checkpoint_dir, args.test_model))

    ###########ACTIVATION FUNCTIONS################
    def tanh(self, X):
        return tf.nn.tanh(X)

    def sigmoid(self, X):
        return tf.nn.sigmoid(X)

    def relu(self, X):
        return tf.nn.relu(X)

    def softmax(self, X):
        return tf.nn.softmax(X)

    def softmaxth(self, X):
        """
        Used for final-act == tanh and loss == cross-entropy
        :param X:
        :return:
        """
        return tf.nn.softmax(tf.nn.tanh(X))

    def linear(self, X):
        return X

    ########LOSS FUNCTION################
    def cross_entropy(self, yhat):
        return tf.reduce_mean(-tf.log(tf.diag_part(yhat)))

    def bpr(self, yhat):
        yhatT = tf.transpose(yhat)
        return tf.reduce_mean(-tf.log(tf.nn.sigmoid(tf.diag_part(yhat) - yhatT)))

    def top1(self, yhat):
        yhatT = tf.transpose(yhat)
        term1 = tf.reduce_mean(tf.nn.sigmoid(-tf.diag_part(yhat) + yhatT) + tf.nn.sigmoid(yhatT ** 2), axis=0)
        return tf.reduce_mean(term1)

    def bpr_max(self, yhat):
        softmax_score = self.softmax(yhat)
        return tf.reduce_mean(-tf.log(tf.reduce_sum(tf.nn.sigmoid(tf.diag_part(yhat) - yhat) * softmax_score, axis=1)) + self.weight_decay*tf.reduce_sum(softmax_score*yhat*yhat, axis=1))

    def top1_max_decay(self, yhat):
        yhatT = tf.transpose(yhat)
        softmax_score = self.softmax(yhat)
        term1 = tf.reduce_sum(softmax_score * (tf.nn.sigmoid(-tf.diag_part(yhat) + yhatT) + self.weight_decay*tf.nn.sigmoid(yhatT ** 2)), axis=1)
        return tf.reduce_mean(term1)
    ################BUILD MODEL###################
    def build_model(self):
        self.X = tf.placeholder(tf.int32, [self.batch_size], name='input')
        self.Y = tf.placeholder(tf.int32, [self.batch_size], name='output')
        self.state = [tf.placeholder(tf.float32, [self.batch_size, self.rnn_size]) for _ in range(self.layers)]
        initializer = tf.random_uniform_initializer(minval=-0.95, maxval=0.95)

        W = tf.get_variable('W', [self.n_items, self.rnn_size], initializer=initializer)
        b = tf.get_variable('b', [self.n_items], initializer=tf.constant_initializer(0.0))

        cell = tf.contrib.rnn.GRUCell(self.rnn_size, activation=self.hidden_act)
        drop_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_p_hidden)
        stacked_cell = tf.contrib.rnn.MultiRNNCell([drop_cell] * self.layers)

        input = tf.one_hot(self.X, depth=self.n_items)

        output, state = stacked_cell(input, tuple(self.state))
        self.final_state = state

        if self.is_training:
            sampled_W = tf.nn.embedding_lookup(W, self.Y)
            sampled_b = tf.nn.embedding_lookup(b, self.Y)
            logits = tf.matmul(output, sampled_W, transpose_b=True) + sampled_b
            self.yhat = self.final_activation(logits)
            self.cost = self.loss_function(self.yhat)
            if self.optimize == 'Adam':
                self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
            elif self.optimize == 'RMS':
                self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)
            elif self.optimize =='GD':
                self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
        else:
            logits = tf.matmul(output, W, transpose_b=True) + b
            self.yhat = self.final_activation(logits)
            # self.cost = self.cross_entropy_test_loss(self.yhat)

    def init(self, data):
        data.sort_values([self.session_key, self.time_key], ascending=True)
        offset_sessions = np.zeros(data[self.session_key].nunique() + 1, dtype=np.int32)
        offset_sessions[1:] = data.groupby(self.session_key).size().cumsum()
        return offset_sessions

    def fit(self, data):
        self.error_during_train = False
        itemids = data[self.item_key].unique()  # id cac item khac nhau
        self.n_items = len(itemids)  # so luong cac item
        self.itemidmap = pd.Series(data=np.arange(self.n_items),
                                   index=itemids)  # chuyen itemid thanh so tuong ung tu 1-> n_items
        data = pd.merge(data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}),
                        on=self.item_key, how='inner')
        data = data.sort_values([self.session_key, self.time_key], ascending=True)
        offset_sessions = self.init(data)  # session offset la array cac index

        print('fitting model...')
        for epoch in range(self.n_epochs):
            epoch_cost = []
            state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in range(self.layers)]
            session_idx_arr = np.arange(len(offset_sessions) - 1)  # session index
            iters = np.arange(self.batch_size)  # so luong batch_size
            maxiter = iters.max()
            start = offset_sessions[session_idx_arr[iters]]
            end = offset_sessions[session_idx_arr[iters] + 1]
            finished = False
            while not finished:
                minlen = (end - start).min()
                out_idx = data.ItemIdx.values[start]
                for i in range(minlen - 1):
                    in_idx = out_idx
                    out_idx = data.ItemIdx.values[start + i + 1]
                    # prepare inputs, targeted outputs and hidden states
                    fetches = [self.cost, self.final_state, self.train_op]
                    feed_dict = {self.X: in_idx, self.Y: out_idx}
                    for j in range(self.layers):
                        feed_dict[self.state[j]] = state[j]

                    cost, state,  _ = self.sess.run(fetches, feed_dict)
                    epoch_cost.append(cost)
                    if np.isnan(cost):
                        print(str(epoch) + ':Nan error!')
                        self.error_during_train = True
                        return

                start = start + minlen - 1
                mask = np.arange(len(iters))[(end - start) <= 1]
                for idx in mask:
                    maxiter += 1
                    if maxiter >= len(offset_sessions) - 1:
                        finished = True
                        break
                    iters[idx] = maxiter
                    start[idx] = offset_sessions[session_idx_arr[maxiter]]
                    end[idx] = offset_sessions[session_idx_arr[maxiter] + 1]
                if len(mask):
                    for i in range(self.layers):
                        state[i][mask] = 0
            avgc = np.mean(epoch_cost)
            print('Epoch {}\tloss: {:.6f}'.format(epoch, avgc))

            avgc = np.mean(epoch_cost)
            if np.isnan(avgc):
                print('Epoch {}: Nan error!'.format(epoch, avgc))
                self.error_during_train = True
                return
            self.saver.save(self.sess, '{}/gru-model'.format(self.checkpoint_dir), global_step=epoch)

    def predict_next_batch(self, session_ids, input_item_ids, itemidmap, batch=100):
        if batch != self.batch_size:
            raise Exception('Predict batch size({}) must match train batch size({})'.format(batch, self.batch_size))
        if not self.predict:
            self.current_session = np.ones(batch) * -1
            self.predict = True

        session_change = np.arange(batch)[session_ids != self.current_session]
        if len(session_change) > 0:  # change internal states with session changes
            for i in range(self.layers):
                self.predict_state[i][session_change] = 0.0
            self.current_session = session_ids.copy()

        in_idxs = itemidmap[input_item_ids]
        fetches = [self.yhat, self.final_state]
        feed_dict = {self.X: in_idxs}
        for i in range(self.layers):
            feed_dict[self.state[i]] = self.predict_state[i]
        preds, self.predict_state = self.sess.run(fetches, feed_dict)
        preds = np.asarray(preds).T
        return pd.DataFrame(data=preds, index=itemidmap.index)