
# 导入包
import pandas as pd
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

import jieba
import re

from sklearn.neighbors import KNeighborsRegressor

# 加载数据

file = 'jobs_4k.xls'
content = pd.read_excel(file)
print(content)

# Part2 使用Network 可视化
position_names = content['positionName'].tolist()
skill_labels = content['skillLables'].tolist()

skill_position_graph = defaultdict(list)
for p,s in zip(position_names, skill_labels):
    skill_position_graph[p] += eval(s)


skill_position_graph = defaultdict(list)
for p,s in zip(position_names, skill_labels):
    skill_position_graph[p] += eval(s)

G = nx.Graph(skill_position_graph)
# 设置中文字体为黑体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# 以10个随机选取的工作岗位为例
sample_nodes = random.sample(position_names, k=10)
sample_nodes

# 初始化节点（随机的30个职位）
sample_nodes_connections = sample_nodes
# 给随机的2个职位，添加相关的技能
for p, skills in skill_position_graph.items():
    if p in sample_nodes:
        sample_nodes_connections += skills

# 抽取G中的子图
sample_graph = G.subgraph(sample_nodes_connections)
plt.figure(figsize=(50, 30))
pos = nx.spring_layout(sample_graph, k=1)
nx.draw(sample_graph, pos, with_labels=True, node_size=30, font_size=10)
plt.show()

# PageRank算法对核心能力和核心职位进行影响力的排序
pr = nx.pagerank(G, alpha=0.9)
ranked_position_and_ability = sorted([(name, value) for name, value in pr.items()], key=lambda x:x[1], reverse=True)
ranked_position_and_ability[:5]

# 特征X 需要去掉salary字段
X_content = content.drop(['salary'], axis=1)
# 目标Target
target = content['salary'].tolist()

# 将X_content内容都拼接成字符串，设置为merged字段
X_content['merged'] = X_content.apply(lambda x: ''.join(str(x)), axis=1)
X_content['merged'][3]

# 合并
def get_one_row_job_string(x_string_row):
    job_string = ''
    for i, element in enumerate(x_string_row.split('\n')):
        if len(element.split()) == 2:
            _, value = element.split()
            job_string += value
    return job_string

# 正则表达式
def token(string):
    return re.findall('\w+', string)
X_string = X_content['merged'].tolist()

cutted_X = []
for i, row in enumerate(X_string):
    job_string = get_one_row_job_string(row)
    cutted_X.append(' '.join(list(jieba.cut(' '.join(token(job_string))))))

#Part 3
# 使用TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cutted_X)

target_numical = [np.mean(list(map(float, re.findall('\d+', s)))) for s in target]
Y = target_numical

# Part4 KNN回归，训练能力和薪资匹配模型

model = KNeighborsRegressor(n_neighbors=2)
model.fit(X, Y)

# Step5 预测薪资
def predict_by_label(test_string, model):
    # 分词
    test_words = list(jieba.cut(test_string))
    # 转换为TF-IDF向量
    test_vec = vectorizer.transform(test_words)
    # 模型预测
    y_pred = model.predict(test_vec)
    return y_pred[0]

test = '测试 北京 3年 专科'

predict_by_label(test, model)

test_list = ['测试 北京 3年 专科',
             '测试 北京 4年 专科',
             '算法 北京 4年 本科',
             'UI 北京 4年 本科',
             "广州Java本科3年掌握大数据",
             "沈阳Java硕士3年掌握大数据", 
             "沈阳Java本科3年掌握大数据", 
             "北京算法硕士3年掌握图像识别"]

for test in test_list:
    print(test, "预测薪资为：", predict_by_label(test, model), "K")
"""