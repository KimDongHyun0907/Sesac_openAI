# https://swexpertacademy.com/main/learn/course/lectureProblemViewer.do

n=int(input())

def perm(depth, acc):
    global ans, check

    if acc >= ans:
        return
    
    if depth==m:
        ans=min(ans, acc)
        return
    
    for i in range(m):
        if not check & (1<<i):
            check |= (1<<i)
            perm(depth+1, acc+arr[depth][i])
            check &= ~(1<<i)


for i in range(1, n+1):
    m=int(input())
    arr = [list(map(int,input().split())) for _ in range(m)]
    ans = 987654321
    check=0
    perm(0,0)
    print(f'#{i} {ans}')