import test

path="C:/Users/bluecom014/Desktop/sesac_openai/AI_활용_서비스_개발/py/txt/"
file = 'val.txt'

with open(path+file,'r') as f:
    content = f.readlines()

test.prn(content)
