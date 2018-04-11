# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse

import model
import evaluate

# PATH_TO_TRAIN = './data-3mi/rsc15_train_full.txt'
PATH_TO_TRAIN = './data-3mi/rsc15_test.txt'
PATH_TO_TEST = './data-3mi/rsc15_test.txt'


class Args():
    is_training = False
    n_epochs = 10
    batch_size = 50
    layers = 1
    rnn_size = 100
    test_model = 1
    dropout_p_hidden = 0.8
    learning_rate = 0.005
    sigma = 0
    init_as_normal = False
    reset_after_session = True
    session_key = 'SessionId'
    item_key = 'ItemId'
    time_key = 'Time'
    grad_cap = 0
    checkpoint_dir = './checkpoint_cr_50_dt_3mi'
    loss = 'cross-entropy'
    final_act = 'softmax'
    hidden_act = 'tanh'
    n_items = -1


def parseArgs():
    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--layer', default=1, type=int)
    parser.add_argument('--size', default=100, type=int)
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--test', default=1, type=int)
    parser.add_argument('--hidden_act', default='tanh', type=str)
    parser.add_argument('--final_act', default='softmax', type=str)
    parser.add_argument('--loss', default='cross-entropy', type=str)
    parser.add_argument('--dropout', default='0.5', type=float)
    parser.add_argument('--checkpoint_dir', default='dir', type=str)

    return parser.parse_args()


def main():
    command_line = parseArgs()
    data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId': np.int64})
    valid = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId': np.int64})
    args = Args()
    args.n_items = len(data['ItemId'].unique())

    args.layers = command_line.layer
    args.rnn_size = command_line.size
    args.n_epochs = command_line.epoch
    args.learning_rate = command_line.lr
    args.is_training = command_line.train
    args.test_model = command_line.test
    args.hidden_act = command_line.hidden_act
    args.final_act = command_line.final_act
    args.loss = command_line.loss
    args.checkpoint_dir = command_line.checkpoint_dir
    args.dropout_p_hidden = 1.0 if args.is_training == 0 else command_line.dropout
    print(args)

    with tf.Session() as sess:
        gru = model.GRU4Rec(sess, args)
        if args.is_training:
            print("Traing only")
            gru.fit(data)
    #     else:
    #         print('EVALUATE:')
    #         res = evaluate.evaluate_sessions_batch(gru, data, valid, batch_size=50)
    #         print('Recall@20: {}\tMRR@20: {}'.format(res[0], res[1]))
    if not args.is_training:
        result = []
        for i in range(0,args.n_epochs): #number epoch
            tf.reset_default_graph()
            with tf.Session() as eval_sess:

                args.test_model = i
                gru = model.GRU4Rec(eval_sess, args)
                res = evaluate.evaluate_sessions_batch(gru, data, valid, batch_size=args.batch_size)
                print('Epoch {}\tRecall@20: {}\tMRR@20: {}\tLoss: {}'.format(i,res[0], res[1], res[2]))
                result.append(res)
        with open(args.checkpoint_dir+'_result_'+str(args.loss)+'_'+str(args.batch_size)+'.txt','w') as file:
            for rs in result:
                file.write('{}\t{}\n'.format(rs[0], rs[1]))
        # print(result)


if __name__ == '__main__':
    main()