from decimal import Decimal as D, getcontext
getcontext().prec = 80
anchor = [(1152.068,25.670945),(1174.084,25.723668),(1207.220,25.767877),(1253.696,25.822174),(1326.328,25.884390),(1569.352,26.307338)]
test = [(1151.356,25.697093),(1175.428,25.747508),(1207.796,25.794290),(1251.228,25.841255),(1326.504,25.908804),(1568.004,26.298732)]
def fit(points):
    xy=[(D(str(ps)),D(str(br)).ln()) for br,ps in points]
    m=[[sum(x**(i+j) for x,_ in xy) for j in range(4)]+[sum(y*x**i for x,y in xy)] for i in range(4)]
    for i in range(4):
        pivot=max(range(i,4),key=lambda k:abs(m[k][i]))
        m[i],m[pivot]=m[pivot],m[i]
        factor=m[i][i]
        m[i]=[v/factor for v in m[i]]
        for k in range(4):
            if k != i:
                factor=m[k][i]
                m[k]=[v-factor*w for v,w in zip(m[k],m[i])]
    return [r[-1] for r in m]
ca,cb=fit(anchor),fit(test)
lo=max(min(D(str(p[1])) for p in anchor),min(D(str(p[1])) for p in test))
hi=min(max(D(str(p[1])) for p in anchor),max(D(str(p[1])) for p in test))
integral=sum((cb[i]-ca[i])*(hi**(i+1)-lo**(i+1))/D(i+1) for i in range(4))
print(((integral/(hi-lo)).exp()-1)*100)
