import sys
input = sys.stdin.readline

n=int(input())
ans = 0

check_vertical = [False]*n
check_diagnal_right = [False]*(2*n-1)
check_diagnal_left = [False]*(2*n-1)

def n_queen(depth):
    global ans

    if depth==n:
        ans += 1
        return
    
    for idx in range(n):
        if check_vertical[idx] or check_diagnal_left[depth+idx] or check_diagnal_right[depth-idx]:
            continue

        check_vertical[idx]=1
        check_diagnal_right[depth-idx]=1
        check_diagnal_left[depth+idx]=1

        n_queen(depth+1)

        check_vertical[idx]=0
        check_diagnal_right[depth-idx]=0
        check_diagnal_left[depth+idx]=0

n_queen(0)
print(ans)  