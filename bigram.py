import random

class Bigram:
    def __init__(self) -> None:
        self.bos = '<BOS>'
        self.eos = '<EOS>'

        self.bigrams = {}
        self.starts = {}
        pass
    
    def get_bigram_count(self, bigram: tuple):
        return self.bigrams.get(bigram, 0)

    def inc_bigram(self, bigram: tuple, cnt: int = 1):
        self.bigrams[bigram] = self.bigrams.get(bigram, 0) + 1

    def get_start_count(self, start: str):
        return self.starts.get(start, 1) # 避免除以0的情况
    
    def inc_starts(self, start: str, cnt: int = 1):
        self.starts[start] = self.starts.get(start, 0) + 1

    def train(self, corpus: list):
        """
            params: \n
            \t corpus: list(str) ["今天 天气 真不错", "中国 国家 广播 电台"]
            return: None
        """
        for line in corpus:
            line = [self.bos] + line.split(" ") + [self.eos]
            # line : ["<BOS>", "今天", "天气", "真不错" ,"<EOS>"]
            pres = line[:-1]
            afters = line[1:]

            for pre, after in zip(pres, afters):
                self.inc_bigram((pre, after), 1)
                self.inc_starts(pre, 1)
            
    # 规定将要消岐的句子，其格式为字符串的列表
    def get_prob(self, text: list):
        """
            test: list(str) ["今天", "天气", "真不错"]
            return: float
        """
        prob = 100 # 避免概率过小
        text = [self.bos] + text + [self.eos]
        pres = text[:-1]
        afters = text[1:]

        details = []

        for pre, after in zip(pres, afters):
            prob *= self.get_bigram_count((pre, after))/self.get_start_count(pre)
            # print(f"({pre}, {after}): {self.get_bigram_count((pre, after))}" )
            # print(f"{pre}: {self.get_start_count(pre)}" )
            detail = {
                (pre, after) : self.get_bigram_count((pre, after)),
                pre: self.get_start_count(pre)
            }
            details.append(detail)

        return prob, details

    # 消岐，将概率最大的句子返回
    def smooth(self, numerator, denominator):
        # total = sum([cnt for token, cnt in self.starts.items()]) + 1 
        total = len(self.starts) + 1
        return (1 + numerator)/total
    
    def get_prob_smooth(self, text: list):
        """
            text: list(str) ["今天", "天气", "真不错"]
            return: float
        """
        prob = 1e150
        text = [self.bos] + text + [self.eos]
        pres = text[:-1]
        afters = text[1:]

        details = []

        for pre, after in zip(pres, afters):
            prob *= self.smooth(self.get_bigram_count((pre, after)), self.get_start_count(pre))
            detail = {
                (pre, after) : self.get_bigram_count((pre, after)),
                pre: self.get_start_count(pre)
            }
            details.append(detail)

        return prob, details
    
    def disambiguation(self, results: list, smooth=False):
        """
            params:\n
            \t  results: list(list(str))\n
            return: list(str)
        """
        prob_func = self.get_prob if not smooth else self.get_prob_smooth

        target = results[0]
        prob = 0
        for res in results:
            target = res if prob_func(res)[0] > prob else target
            prob = prob_func(res)[0] if prob_func(res)[0] > prob else prob
        
        # 如果所有分词结果都因稀疏而产生零概率，随机选择
        if prob == 0:
            i = random.randint(0, len(results)-1)
            target = results[i]


        return target


