# https://www.acmicpc.net/problem/11723

import sys
input = sys.stdin.readline

n=int(input())
s=0
for _ in range(n):
    commands = input().split()

    if len(commands)==1:
        if commands[0] == 'all':
            s=(1<<21)-1
            
        elif commands[0] == 'empty':
            s=0

    else:
        command, num = commands[0], int(commands[1])
        if command == 'add':
            s |= (1<<num)
        if command == 'check':
            print(0 if s&(1<<num)==0 else 1)
        if command == 'remove':
            s &= ~(1<<num)
        if command == 'toggle':
            s ^= (1<<num)