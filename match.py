from pypinyin import lazy_pinyin
import re

class Match:
    def __init__(self, method: str, vocab: list):
        """
            用于初始化的vocab为字符串的列表 \n
            格式如下： \n
            vocab: ["今天", "天气", "真不错"]
        """

        self.__methods = {
            'max_forward': self.__max_forward,
            'max_backward': self.__max_backward,
        }
        self.__vocab = sorted(vocab, key=lambda x: ''.join(lazy_pinyin(x)))
        self.__max_word_len = 0
        self.tokenize = self.__methods[method]
        for word in self.__vocab:
            if len(word) > self.__max_word_len:
                self.__max_word_len = len(word)
        self.__vocab_len = len(self.__vocab)
        self.__vocab_set = set(vocab)
        
        
    def __max_forward(self, sentence: str, search_type: str = "set"):
        index = 0 # 当前移动的指针
        res = []
        
        length = self.__max_word_len
        if len(sentence) < length:
            length = len(sentence)
        
        while index < len(sentence):
            # 如果多字词都未匹配成功，则将字放入结果
            target = sentence[index]
            cur = index
            # 判断可以达到的最大下标是否超过末尾
            if index + length < len(sentence):
                rear = length
            else:
                rear = len(sentence)
            
            # 最大前向匹配
            for i in range(rear, 1, -1):
                if self.__search_word(sentence[index:index+i], search_type):
                    target = sentence[index:index+i]
                    index += i
                    break
            # if target == sentence[index]:
            #     index += 1
            if cur == index:
                index += 1
            res.append(target)
        return res
                
    def __max_backward(self, sentence: str, search_type: str = "set"):
        index = len(sentence) # 当前移动的指针
        res = []
        
        length = self.__max_word_len
        if len(sentence) < length:
            length = len(sentence)
        
        while index > 0:
            # 如果多字词都未匹配成功，则将字放入结果
            target = sentence[index-1]

            # 判断可以达到的最大下标是否超过开头
            if index-length >= 0:
                front = length
            else:
                front = index
            
            # 最大前向匹配
            for i in range(front, 1, -1):
                if self.__search_word(sentence[index-i:index], search_type):
                    target = sentence[index-i:index]
                    index -= i
                    break
            if target == sentence[index-1]:
                index -= 1
            res = [target] + res
        return res

    def __max_forward_recur(self, sentence: str, search_type: str = "set"):
        if len(sentence) == 0:
            return list()
        target = sentence[0]
        length = self.__max_word_len
        if len(sentence) < length:
            length = len(sentence)
        sub = sentence[:length]

        #窗口中循环匹配
        for i in range(length, 0, -1):
            if self.__search_word(sub[:i], search_type):
                target = sub[:i]
                break
        #递归
        res = self.__max_forward(sentence[len(target):])
        res = [target] + res
        return res

    def __max_backward_recur(self, sentence: str, search_type: str = "set"):
        if len(sentence) == 0:
            return list()
        target = sentence[-1]
        length = self.__max_word_len
        if len(sentence) < length:
            length = len(sentence)
        sub = sentence[-length:]

        for i in range(length):
            if self.__search_word(sub[i:], search_type):
                target = sub[i:]
                break   
        res = self.__max_backward(sentence[:-len(target)])
        res.append(target)
        return res
    
    def __is_in_vocab(self, word: str) -> bool:
        left = 0
        right = len(self.__vocab)-1
        word_pinyin = ''.join(lazy_pinyin(word))

        # 如果字符串中包含非字母，表明有标点符号，返回False
        pattern = r'^[a-zA-Z]+$'
        if not re.match(pattern, word_pinyin):
            return False    
        
        while left <= right:
            mid = (left + right)//2
            vocab_word_pinyin = ''.join(lazy_pinyin(self.__vocab[mid]))
            if vocab_word_pinyin == word_pinyin:
                # 如果出现拼音相同，为处理拼音相同字不同的情况，直接在该位置附近顺序扫面
                if word in self.__vocab[(mid-10+self.__vocab_len)%self.__vocab_len:(mid+10)%self.__vocab_len]:
                    return True
                else:
                    return False
            elif vocab_word_pinyin < word_pinyin:
                left = mid+1
            else:
                right = mid-1

        return False
    
    def __search_word(self, word: str, search_type: str = None) -> bool:
        if search_type is None:
            return word in self.__vocab
        elif search_type == 'binary_search':
            return self.__is_in_vocab(word)
        elif search_type == 'set':
            return (word in self.__vocab_set)

