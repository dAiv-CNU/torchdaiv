from typing import Union
import numpy as np
from torch.utils.data import Dataset


class FixedLengthCalculatorDataset(Dataset):
    operators = ['+', '+', '+', '-', '-', '-', '*', '/']

    def __init__(self, size: int = 10000, max_length: Union[int, int] = [3, 4]):
        self.size = size
        self.max_length = max_length
        self.data, self.labels = zip(*[self.generate_data(max_length) for _ in range(size)])

    @classmethod
    def generate_data(cls, max_length: Union[int, int] = [3, 4]):
        # 무작위로 숫자와 연산자 생성
        num1 = np.random.randint(0, 10**max_length[0]-1)
        num2 = np.random.randint(0, 10**max_length[0]-1)
        op = np.random.choice(cls.operators).item()
        if op == '/' and num2 == 0:  # Zero Division Error 방지
            num2 += np.random.randint(1, 10)

        # 숫자를 각 자릿수별로 분리하여 ASCII 변환
        num1_digits = (['\00'] * max_length[0])  # 패딩 토큰 (NUL)
        num2_digits = (['\00'] * max_length[0])
        num1_digits[-len(str(num1)):] = str(num1)
        num2_digits[-len(str(num2)):] = str(num2)

        # 입력 벡터 생성 (num1의 각 자릿수 + operator + num2의 각 자릿수)
        input_arr = num1_digits + [op] + num2_digits
        output_str = str(eval("".join(input_arr).replace("\00", "")))

        # 결과 벡터 생성
        output_arr = (['\00'] * max_length[1])
        if len(output_str) > max_length[1]:
            try:
                int(output_str)
                output_arr[:len(output_str)] = output_str[-max_length[1]:]
            except:  # float
                output_arr[:len(output_str)] = output_str[:max_length[1]]
        else:
            output_arr[:len(output_str)] = output_str
        return input_arr, output_arr

    def __len__(self):
        return self.size

    def sample(self, idx):
        joined = "".join(self.data[idx]), "".join(self.labels[idx])
        out = eval(joined[0].replace("\00", ""))
        print(f"[[DATA {idx}]] >>>", joined[0], self.data[idx], "==", joined[1], self.labels[idx], f" <REAL VAL: {out}>")

    def __getitem__(self, idx):
        # Min-Max 정규화 (ASCII 코드 범위: 0-127 사용)
        norm = lambda input_ascii: [x / 127 for x in map(ord, input_ascii)]
        return norm(self.data[idx]), norm(self.labels[idx])
