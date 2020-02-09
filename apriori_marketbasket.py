from efficient_apriori import apriori

# 数据加载
data = open('./Market_Basket_Optimisation.csv')
transactions = []
temp_set = set()
#逐行读取
for line in data.readlines():
    line_1 = line.rstrip()
    line_new = line_1.split(',')#type of line_new is list

    temp_set.clear()
    for item in line_new:
        temp_set.add(item)
    #print(temp_set)
    # 将数据集进行格式转换
    transactions.append(temp_set)
    #print(transactions)

# 挖掘频繁项集和频繁规则
itemsets, rules = apriori(transactions, min_support=0.02,  min_confidence=0.5)
print('频繁项集：', itemsets)
print('关联规则：', rules)


