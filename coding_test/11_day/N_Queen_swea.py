n=int(input())

def n_queen(depth):
    global ans

    if depth==m:
        ans+=1
        return
    
    for i in range(m):
        if check_vertical[i] or check_diagnal_right[depth-i] or check_diagnal_left[depth+i]:
            continue

        check_vertical[i], check_diagnal_right[depth-i], check_diagnal_left[depth+i] = 1,1,1
        n_queen(depth+1)
        check_vertical[i], check_diagnal_right[depth-i], check_diagnal_left[depth+i] = 0,0,0

for i in range(1, n+1):
    m=int(input())
    ans=0
    check_vertical = [False for _ in range(m)]
    check_diagnal_right = [False for _ in range(2*m+1)]
    check_diagnal_left = [False for _ in range(2*m+1)]
    n_queen(0)
    print(f'#{i} {ans}')