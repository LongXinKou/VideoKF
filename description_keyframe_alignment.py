original_list = [1, 1, 1, 2, 0, 0]

# 使用 sorted() 函数对列表进行排序
sorted_list = sorted(original_list)

# 创建一个新的列表，用于存储排序后的元素
result_list = []

# 遍历排序后的列表
for num in sorted_list:
    # 计算原始列表中有多少个相同的元素
    count = original_list.count(num)
    
    # 将相同元素添加到结果列表中
    result_list.extend([num] * count)

print(result_list)