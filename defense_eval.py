import os
import pandas as pd


def load_labels(file_name):
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: int(dev.iloc[i]['label']) for i in range(len(dev))}
    return f2l


def load_result_from_txt(file_name):
    reslut = {}
    with open(file_name, 'r') as fp:
        for line in fp.readlines():
            img_name, label = line.strip().split(',')
            # print(img_name, label)
            reslut[img_name] = int(label)
    return reslut



if __name__ == '__main__':
    f2l = load_labels('./dev_data/val_rs.csv')
    reslut = load_result_from_txt('./nips_ai_si_ti_di_ens.txt')
    # print(reslut)
    acc = 0
    for item in reslut:
        if reslut[item] == f2l[item]:
            acc += 1
    
    print(acc)
    print(100 - (acc / len(reslut)) * 100)