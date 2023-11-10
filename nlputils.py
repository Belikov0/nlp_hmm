import matplotlib.pyplot as plt
import numpy as np
import random

def show_plot(x, y, x_label, y_label):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axhline(np.mean(y), color='r', linestyle='--', label='avg')
    plt.fill_between(x, np.mean(y)-np.std(y), np.mean(y)+np.std(y), alpha=0.3, label='variance area')
    plt.ylim(0.5, 1)

    plt.legend()
    plt.show()

def read_corpus_or_states_for_hmm(file_path, valid_size = 0.2):
    """
        将文件处理成list(str) \n
        列表里每一行是一句分词好的句子 \n
        return: list(str) ["今天 天气 真不错", "中国 国家 广播 电台"]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        corpus_file = f.read().split('\n')
    return corpus_file

def yield_data(dataset, valid_size: float = 0.2, shuffle=True):
    if not valid_size:
        return dataset, None
    
    if not isinstance(dataset, list):
        dataset = list(dataset)

    if shuffle:
        random.shuffle(dataset)  
    
    return dataset[:int(len(dataset)*(1-valid_size))], dataset[int(len(dataset)*(1-valid_size)):]

def k_fold_cross(dataset, k=1, shuffle=True):
    """
        将list数据集或zip(list, list)数据集进行k折划分 \n
        并用yield产生k轮的训练数据和验证数据
    """
    if k == 1:
        return dataset

    if not isinstance(dataset, list):
        dataset = list(dataset)

    if shuffle:
        random.shuffle(dataset)
    
    size = len(dataset)//k

    for i in range(k):
        train_set = dataset[i*size:(i+1)*size]
        test_set = dataset[:i*size] + dataset[(i+1)*size:]
        yield train_set, test_set

def read_vocab_for_match(path):
    """
        return: list(str) ["今天", "天气", "真不错", "中国", "国家", "广播", "电台"] 
    """
    vocab = []
    with open(path, 'r', encoding='utf-8') as f:
        f.readline()
        lines = f.readlines()
        for line in lines:
            vocab.append(line.split(' ')[-1].replace('\n', ''))

    return vocab

def corpus_to_vocab(corpus):
    """
        将形如 ["今天 天气 真不错\\n", "中国 国家 广播 电台\\n"] 格式的语料 \n
        params: \n
        \t corpus: list(str) ["今天 天气 真不错\\n", "中国 国家 广播 电台\\n"]
        return: list(str) ["今天", "天气", "真不错", "中国", "国家", "广播", "电台"]
    """
    vocab = []
    for line in corpus:
        vocab += line.replace('\n', '').split(' ')
    return vocab

def cal_count(real_res, pred_res):
    real_count = len(real_res)
    pred_count = len(pred_res)
    correct_count = 0

    preal = [0]
    ppred = [0]

    for real_word in real_res:
        preal.append(preal[-1]+len(real_word))
    for pred_word in pred_res:
        ppred.append(ppred[-1]+len(pred_word))
    
    real_set = set(zip(ppred[:-1], ppred[1:]))
    for word_indexes in zip(preal[:-1], preal[1:]):
        if word_indexes in real_set:
            correct_count += 1
    
    return real_count, pred_count, correct_count

def cal_prf(real, pred, correct):
    p = correct/real
    r = correct/pred
    f = (2*p*r)/(p+r)
    return p, r, f

    
def two_gram():
    return 