# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import baseline
import evaluate
import argparse
import win_unicode_console
win_unicode_console.enable()

def parseArgs():
    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--data', default='data-3mi',type=str)

    return parser.parse_args()


def main():
    command_line = parseArgs()
    path_to_train = './'+command_line.data+'/rsc15_train_full.txt'
    path_to_test = './' + command_line.data+'/rsc15_test.txt'
    # path_to_train = path_to_test
    # path_to_test = path_to_train
    # print(path_to_train)
    # print(path_to_test)
    data = pd.read_csv(path_to_train, sep='\t', dtype={'ItemId': np.int64})
    valid = pd.read_csv(path_to_test, sep='\t', dtype={'ItemId': np.int64})

    itemKNN_model = baseline.ItemKNN()
    itemKNN_model.fit(data)
    result = evaluate.evaluate_sessions(itemKNN_model, valid, data)
    print('Recall {}, MRR {}'.format(result[0], result[1]))



if __name__ == '__main__':
    main()
