# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import argparse
import ItemKNN
import win_unicode_console
win_unicode_console.enable()

def parseArgs():
    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--data', default='dml10m',type=str)
    return parser.parse_args()

def main():
    command_line = parseArgs()
    path_to_train = './'+command_line.data+'/rsc15_train_full.txt'
    path_to_test = './' + command_line.data+'/rsc15_test.txt'
    # path_to_train = path_to_test
    data = pd.read_csv(path_to_train, sep='\t', dtype={'ItemId': np.int64})
    valid = pd.read_csv(path_to_test, sep='\t', dtype={'ItemId': np.int64})
    print(len(valid['SessionId'].unique()))
    print(len(valid['ItemId'].unique()))
    itemKNN_model = ItemKNN.ItemKNN()
    itemKNN_model.fit(data)
    result = ItemKNN.evaluate_ItemKNN(itemKNN_model, valid, data)
    print('Recall {}, MRR {}'.format(result[0], result[1]))

if __name__ == '__main__':
    main()
