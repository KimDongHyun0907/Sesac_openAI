import test
from pathlib import Path

FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]
# ROOT1 = FILE.parents[1]
# print(FILE)  C:\Users\bluecom014\Desktop\sesac_openai\AI_활용_서비스_개발\py2\module\main.py
# print(ROOT)  C:\Users\bluecom014\Desktop\sesac_openai\AI_활용_서비스_개발\py2\module
# print(ROOT1)  C:\Users\bluecom014\Desktop\sesac_openai\AI_활용_서비스_개발\py2

# path="C:/Users/bluecom014/Desktop/sesac_openai/AI_활용_서비스_개발/py/txt/"

path = str(FILE.parents[1])+'\\txt\\'
file = 'val.txt'

with open(path+file,'r') as f:
    content = f.readlines()

test.prn(content)

# test.prn(content)