import pandas as pd


# bags = [(2,3),(3,4),(4,5),(5,6),(4,3),(7,12),(3,3),(2,2)]
path = './data/large.csv'
bags = pd.read_csv(path)
limit_weight = int(0.5 * sum(bags['weight']))
print(limit_weight)
# limit_weight = 15
length = len(bags)
min_w = min(bags['weight'])
sum_w = sum(bags['weight'])

if limit_weight < min_w:
    print([0] * length)

elif limit_weight >= sum_w:
    print([1] * length)

else:
    blank = [0] * length
    sol = {}

    for j in range(0, limit_weight+1):
        for i in range(length):
            sol.setdefault('_'.join([str(j),str(i)]),[0,blank])

    blank_ = blank
    blank_[0] = 1
    for j in range(1, limit_weight+1):
        if j >= bags['weight'][0]:
            sol['_'.join([str(j),'0'])] = [bags['value'][0],blank_]

    for j in range(1, limit_weight+1):
        for i in range(1, length):
            if bags['weight'][i] > j:
                sol['_'.join([str(j),str(i)])] = sol['_'.join([str(j),str(i-1)])]
            else:
                v1 = sol['_'.join([str(j),str(i-1)])][0]
                v2 = sol['_'.join([str(j-bags['weight'][i]),str(i-1)])][0]+bags['value'][i]
                if v1 >= v2:
                    sol['_'.join([str(j),str(i)])] = sol['_'.join([str(j),str(i-1)])]
                else:
                    index = sol['_'.join([str(j-bags['weight'][i]),str(i-1)])][1][:]
                    index[i] = 1
                    sol['_'.join([str(j),str(i)])] = [v2,index]

ind = sol['_'.join([str(limit_weight),str(length-1)])][1]
print(ind)
print((bags['weight'] * ind).sum())
print((bags['value'] * ind).sum())