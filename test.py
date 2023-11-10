# from hmm import text_to_state, HMM
# from match import Match
# from nlputils import *
# from bigram import Bigram
# import os
# import jieba
# import json

# text_file_path = './corpus/2021.txt'
# dest_folder = './states/'
# states_file_path = dest_folder + os.path.basename(text_file_path)
# # text_to_state(text_file_path, dest_folder)
# corpus = read_corpus_or_states_for_hmm(text_file_path)
# states = read_corpus_or_states_for_hmm(states_file_path)

# train, test = yield_data(zip(corpus, states))
# corpus_train, states_train = zip(*train)
# corpus_test, states_test = zip(*test)

# vocab_train = corpus_to_vocab(corpus_train)
# vocab_train[:10]

b = {
    "a": 1,
    "b": 2
}

print(len(b))
