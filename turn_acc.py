import pandas as pd
import json
# 读取文件1
df1 = pd.read_csv('duo35.tsv', sep='\t', header=None, names=['id', 'caption1', 'caption2'])
# 读取文件2
df2 = pd.read_csv('testduo.tsv', sep='\t', header=None, names=['id', 'json_data'])


df1['caption1_parsed'] = df1['caption1'].apply(lambda x: json.loads(x)[0]['caption'] if isinstance(x, str) else None)
df2['action'] = df2['json_data'].apply(lambda x: json.loads(x)[0]['action'] if isinstance(x, str) else None)
#print(df1[['caption1_parsed']])
#print(df2[['id','action']])

# left准确率
# 检查df1中的caption1_parsed是否包含'left'
df1_contains_left = df1['caption1_parsed'].apply(lambda x: 'left' in str(x) if pd.notna(x) else False)
# 检查df2中的action是否包含'right'
df2_contains_right = df2['action'].apply(lambda x: 'right' in str(x) if pd.notna(x) else False)
overlap = (df1_contains_left & df2_contains_right).sum()

print(f"Overlapping rows_left: {overlap}")
num = df1_contains_left.sum()
print(f"num_left: {num}")
pre= (num-overlap)/num
print(f"pre_left: {pre}")

# right准确率
df1_contains_right = df1['caption1_parsed'].apply(lambda x: 'right' in str(x) if pd.notna(x) else False)
# 检查df2中的action是否包含'right'
df2_contains_left = df2['action'].apply(lambda x: 'left' in str(x) if pd.notna(x) else False)
overlap = (df1_contains_right & df2_contains_left).sum()

print(f"Overlapping rows_right: {overlap}")
num = df1_contains_right.sum()
print(f"num_right: {num}")
pre= (num-overlap)/num
print(f"pre_right: {pre}")