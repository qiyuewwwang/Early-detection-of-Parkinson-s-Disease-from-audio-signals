aaa.py 文件是调用praat来提取语音特征、训练机器学习的模型
因为用的是国外的数据，量比较少 精度只有0.63

pack.py 是自己制作的库，后续的测试要调用

test.py 是调用的示例

audio文件夹 是用来训练的数据
processed_results.csv 是提取出来的特征
trainedModel.sav 是训练后导出的模型

代码里面也有一些批注

aaa.py 里面有logisticRegression、Lasso、SVM （clf，clf2，clf3）
aaa2.py 为最初始代码 只有逻辑回归
pack2.py 用的是svm
