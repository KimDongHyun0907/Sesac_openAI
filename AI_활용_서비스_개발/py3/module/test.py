def prn(val):
    if isinstance(val, str):
        print('문자열 자료형')
        print('-'*100)
        print(val)
    elif isinstance(val, list):
        print('리스트 자료형')
        for x in val:
            print('-'*100)
            print(x)
            