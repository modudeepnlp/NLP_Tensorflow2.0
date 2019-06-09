import re

class Split:
    def __init__(self):
        super(Split, self).__init__()
        self._base_code = 44032
        self._chosung = 588
        self._jungsung = 28
        # 초성 리스트. 00 ~ 18
        self._chosung_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ',
                              'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ',
                              'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        # 중성 리스트. 00 ~ 20
        self._jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ',
                               'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ',
                               'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        # 종성 리스트. 00 ~ 27 + 1(1개 없음)
        self._jongsung_list = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ',
                               'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ',
                               'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    def sentenceSplit(self, x):
        result_list = []
        for sen in x:
            sen_list = []
            str = list(sen)
            for char in str:
                if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', char) is not None:
                    ord_char = ord(char)
                    if ord_char < self._base_code:
                        sen_list.append(char)
                        continue

                    char_code = ord_char - self._base_code
                    alphabet1 = char_code // self._chosung
                    sen_list.append(self._chosung_list[alphabet1])

                    alphabet2 = (char_code - (self._chosung * alphabet1)) // self._jungsung
                    sen_list.append(self._jungsung_list[alphabet2])

                    alphabet3 = (char_code - (self._chosung * alphabet1) - (self._jungsung * alphabet2))
                    if alphabet3 != 0:
                        sen_list.append(self._jongsung_list[alphabet3])
                else:
                    sen_list.append(char)
            result_list.append(sen_list)
        return result_list

# def main():
#     test = "안녕하세요"
#     split_sentence = Split()
#     split_sentence.sentenceSplit(test)
#
#
# if __name__ == "__main__":
#     main()
