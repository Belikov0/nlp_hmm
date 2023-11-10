import numpy as np
import os
from tqdm import tqdm

def get_state_for_word(word: str):
    if len(word) == 1:
        return "S"
    return "B" + "M" * (len(word)-2) + "E"

def text_to_state(file_path: str, output_folder: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    states = ""
    with open(os.path.join(output_folder, os.path.basename(file_path)), 'w', encoding='utf-8') as f:
        for line in tqdm(lines):
            if not line:
                continue
            line_states = ""
            for word in line.split(' '):
                if word != ' ':
                    line_states += get_state_for_word(word) + " "
            line_states += '\n'

            f.write(line_states)


class HMM:
    def __init__(self):
        # self.text = text
        # self.states = states

        self.state_to_index = {"B": 0, "M": 1, "S": 2, "E": 3}
        self.index_to_state = ["B", "M", "S", "E"]
        self.states_length = len(self.state_to_index)

        self.initial_matrix = np.zeros(self.states_length)
        self.transfer_matrix = np.zeros((self.states_length, self.states_length))

        self.emission_matrix = {
            "B": { "total": 0},
            "M": { "total": 0},
            "S": { "total": 0},
            "E": { "total": 0}
        }


    def get_initial_prob(self, state: str):
        return self.initial_matrix[self.state_to_index[state]]

    def get_transfer_prob(self, state: str, after_state: str):
        return self.transfer_matrix[self.state_to_index[state], self.state_to_index[after_state]]

    def get_emit_prob(self, state: str, char: str, human_prob=0):
        return self.emission_matrix[state].get(char, human_prob)

    def train(self, text, states):
        """
            文本序列和状态序列均为分词好的长字符串 \n
            格式如下： \n
            text: "今天 天气 真不错" \n
            states: "BE BE BME" 
        """

        for text_line, states_line in zip(text, states):
            if len(states_line) < 1:
                continue
            text_line = text_line.replace(" ", "")
            states_line = states_line.replace(" ", "")

            prevs = states_line[:-1]
            nexts = states_line[1:]

            # 计算初始矩阵
            self.initial_matrix[self.state_to_index[states_line[0]]] += 1

            #计算发射矩阵
            for char, state in zip(text_line, states_line):
                self.emission_matrix[state][char] = self.emission_matrix[state].get(char, 0) + 1
                self.emission_matrix[state]["total"] += 1

            for prev, ne in zip(prevs, nexts):
                self.transfer_matrix[
                    self.state_to_index[prev],
                    self.state_to_index[ne]
                ] += 1

        self.__normalize()

    def forward(self, text: str):
        assert len(text) > 0
        # 简化变量
        states = self.index_to_state
        indexes = self.state_to_index

        # 定义并初始化向前变量
        alpha = { state: self.get_initial_prob(state) * self.get_emit_prob(state, text[0]) 
                  for state in states }

        # 循环计算
        for t, char in enumerate(text):
            if t == 0:
                continue
            need_extend = False
            cur = {state: 0 for state in states}
            for state in states:
                for pre_state in states:
                    cur[state] += alpha[pre_state] * self.get_transfer_prob(pre_state, state)
                cur[state] *= self.get_emit_prob(state, char)
                # 避免概率消失
                if cur[state] != 0 and cur[state] < 1e-20:
                    need_extend = True
            
            if need_extend:
                cur = {state: prob*10000 for state, prob in cur.items()}
            
            alpha = cur

        res = 0
        for state, prob in alpha.items():
            if state in ["E", "S"]:
                res += prob
        return res 

    def backward(self, text: str):
        # 简化变量
        states = self.index_to_state
        indexes = self.state_to_index

        # 定义向后变量， 句子只能以
        beta = { state: 1 if state in ["E", "S"] else 0 for state in states}

        # 迭代计算向后变量
        for t, char in enumerate(reversed(text)):
            if t == len(text)-1:
                break

            need_extend = False
            cur = {state: 0 for state in states}
            for state in states:
                for after_state in states:
                    cur[state] += beta[after_state] * self.get_transfer_prob(state, after_state) * self.get_emit_prob(after_state, char)
                if cur[state] != 0 and cur[state] < 1e-20:
                    need_extend = True
            
            if need_extend:
                cur = {state: prob*10000 for state, prob in cur.items()}

            beta = cur

        res = 0
        for state, prob in beta.items():
            if state in ["B", "S"]:
                res += prob * self.get_initial_prob(state) * self.get_emit_prob(state, text[0])

        return res

    def tokenize(self, text: str):
        """
            return: list(str) ["今天", "天气", "真不错"] 
        """
        return self.viterbi(text)

    def viterbi(self, text: str):
        if len(text) == 0:
            return "", []

        # 简化变量
        states = self.index_to_state
        indexes = self.state_to_index
        delta = [{}]
        path = {}   # path[state]表示以state状态结尾的路径

        # 初始化，只有B和S可以作为句子的开头
        for state in states:
            delta[0][state] = self.get_initial_prob(state) * self.get_emit_prob(state, text[0]) if state in ["B", "S"] else 0
            path[state] = [state]

        # 遍历字符串
        for t, char in enumerate(text):
            if t == 0:
                continue
            delta.append({})
            cur = {}

            # 判断是否在词表中
            in_char_list = False
            for word_vector in self.emission_matrix.values():
                if char in word_vector.keys():
                    in_char_list = True
                    break

            need_extend = False
            # 循环判断，对每一个state寻找概率最大的前路径
            for state in states:
                emit = self.get_emit_prob(state, char) if in_char_list else 1.0
                # 对当前的state，计算所有可能pre_state的概率并取得最大
                (max_prob, max_state) = max( [(delta[t-1][pre_state] * self.get_transfer_prob(pre_state, state) * emit, pre_state) for pre_state in states] )
                (min_prob, min_state) = max( [(delta[t-1][pre_state] * self.get_transfer_prob(pre_state, state) * emit, pre_state) for pre_state in states] )
                # 将该次判断的结果添加到路径中
                delta[t][state] = max_prob
                cur[state] = path[max_state] + [state]
                # 判断是否概率过小
                if min_prob != 0 and min_prob < 1e-10:
                    need_extend = True
            
            if need_extend:
                delta[t] = {state: prob*100000 for state, prob in delta[t].items()}

            path = cur

        # 在四条路径中选择最大的路径， 只有以E和S结尾的能算作
        (prob, state) = max( [ (delta[len(text)-1][state], state) for state in ["E", "S"] ] )

        # 获得分词
        split = ""

        for char, state in zip(text, path[state]):
            split += char
            if state in ["E", "S"]:
                split += " "
        split = split[:-1].split(" ")
        return path[state], split

    def __normalize(self):
        self.emission_matrix = {state:{
            char: count/word_vector["total"] for char, count in word_vector.items() if char != "total"
        } for state, word_vector in self.emission_matrix.items()}
        self.initial_matrix = self.initial_matrix/np.sum(self.initial_matrix)
        self.transfer_matrix = self.transfer_matrix/np.sum(self.transfer_matrix, axis=1, keepdims=True)