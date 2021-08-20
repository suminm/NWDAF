# 1주일치 data > 그 다음주 월요일을 예측하는 직렬 모델
# Decision Tree, Adaboost, kNN
# train data: Mon~Fri(x) , Tue~Sat(y)
# test data: Wed~Sun(x), Thu~next Mon(y)
# 예측 목표: next Mon 예측하기

# EMA 여부 정할 수 있음.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_squared_error

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import os
from datetime import datetime


data2 = pd.read_csv('data_2w.csv')
data2 = data2.dropna(axis=0)
data3 = pd.read_csv('data_3w.csv')
data3 = data3.dropna(axis=0)
data4 = pd.read_csv('data_4w.csv')
data4 = data4.dropna(axis=0)
df= pd.concat([data2,data3,data4], ignore_index=True)

df = df.iloc[5760:28800]
print(df)


def LPF(f_cut,data):
    w_cut = 2 * np.pi * f_cut
    tau = 1 / w_cut
    ts = 1

    lpf_result = [data[0]]

    for i in range(1, len(data)):
        value = (tau * lpf_result[i - 1] + ts * data[i]) / (ts + tau)
        lpf_result.append(value)

    return(lpf_result)

### x, y define ###
y = df['Energy consumption per timeslot [kWh]'].to_numpy()

# LPF 여부 선택
lpf = input("Low Pass Filtering (EMA) 하시겠습니까? (y/n):")

while lpf != 'y' and lpf != 'n':
    print('다시 입력하십시오.')
    lpf = input("Low Pass Filtering (EMA) 하시겠습니까? (y/n):")

if lpf == 'y':
    f_cut = input('cutting frequency를 입력하세요 (권장=0.02): ')  # 차단주파수 (Hz)
    f_cut = float(f_cut)
    y = np.array(LPF(f_cut, y))

elif lpf == 'n':
    print('Original Data를 사용하겠습니다.')

y_test = y[4320 : 23000]
y_test_cut = y_test[5760:7200]

x_train = y[:18720].reshape(-1, 1)
y_train = y[1440:20160]
x_test = y[2880:21600].reshape(-1, 1)
x_test_cut = x_test[5760:7200]


# 알고리즘의 Parameter 값 바꾸기
change = input("Default 설정 값을 바꾸시겠습니까?(y/n):")
while ((change != 'y') and (change !='n')):
    print('다시 입력하십시오.')
    change = input("Default 설정 값을 바꾸시겠습니까?(y/n):")

if change == 'y':
    # decision tree sampling number 정하기
    decisionsample = input ("Sampling number of Decision Tree (권장=400):")
    decisionsample = int(decisionsample)
    while (decisionsample <= 0):
        print(" 0 이상 자연수를 입력하세요.")
        maxdepth = input("Sampling number of Decision Tree (권장=400):")

    # decision tree max depth 정하기
    maxdepth = input ("Max depth of Decision Tree (if 0, default=None):")
    maxdepth = int(maxdepth)
    while ((maxdepth < 1) and (maxdepth !=0)):
        print(" 1 이상 자연수를 입력하세요.")
        maxdepth = input("Max depth of Decision Tree (if 0, default=None):")
    if maxdepth == 0:
        maxdepth = None

    # adaboost sampling number 정하기
    adaboostsample = input("Sampling number of Adaboost (권장=400):")
    adaboostsample = int(adaboostsample)
    while (adaboostsample < 0) :
        print("0 이상 자연수를 입력하세요.")
        adaboostsample = input("Sampling number of Adaboost (권장=400):")

    # adaboost learning rate 정하기
    learningrate = input("Learning rate of Adaboost (0<learning rate<1) (권장=0.1):")
    learningrate = float (learningrate)
    while ((learningrate < 0) or (learningrate > 1)):
        print("0과 1 사이의 실수를 입력하세요.")
        learningrate = input("Learning rate of Adaboost (0<learning rate<1) (권장=0.1):")

elif change == 'n':
    decisionsample=400
    maxdepth = None
    adaboostsample=400
    learningrate=0.1

repeatnum = input("알고리즘 반복 횟수를 입력하세요:")
repeatnum = int(repeatnum)

###Adaboost###

df2 = pd.DataFrame()
adaboosttrain_list=[]
adaboostRMSE_list=[]
for repeat in range(0,repeatnum):

    adaboost = make_pipeline(StandardScaler(), AdaBoostRegressor(base_estimator=GradientBoostingRegressor(min_samples_split=2, loss='ls', n_estimators=adaboostsample, learning_rate=learningrate, random_state=None))).fit(x_train, y_train)

    if repeat<repeatnum:
        predict_adaboost = adaboost.predict(x_test)
        df2['predict' + str(repeat + 1)] = predict_adaboost
        adaboosttrain_list.append(adaboost.score(x_train, y_train))
        adaboostRMSE_list.append(mean_squared_error(y_test_cut, predict_adaboost[5760:7200], squared=False))

    print(str(repeat + 1) + ' round Ended!')

# 평균 train score
adaboosttrain_list=np.array(adaboosttrain_list)
adaboosttrain=np.mean(adaboosttrain_list)

# 평균 RMSE
adaboostRMSE_list=np.array(adaboostRMSE_list)
adaboostRMSE=np.mean(adaboostRMSE_list)

# 평균 결과
df2['mean'] = df2.iloc[:, 0:repeatnum].mean(axis=1)

print('################################################')

###Decision Tree###
df3=pd.DataFrame()

DTRMSE_list=[]
DTtrain_list=[]

for repeat in range(0,repeatnum):
    decision_tree_model = DecisionTreeRegressor(random_state=None, max_depth=maxdepth)
    decisiontree = BaggingRegressor(base_estimator=decision_tree_model, n_estimators=decisionsample, verbose=0).fit(x_train,y_train)

    if repeat<repeatnum:
        predict_DT=decisiontree.predict(x_test)
        df3['predict' + str(repeat + 1)] = predict_DT
        DTtrain_list.append(decisiontree.score(x_train, y_train))
        DTRMSE_list.append(mean_squared_error(y_test_cut, predict_DT[5760:7200], squared=False))

    print(str(repeat + 1) + ' round Ended!')

# 평균 train score
DTtrain_list=np.array(DTtrain_list)
DTtrain=np.mean(DTtrain_list)

# 평균 RMSE
DTRMSE_list=np.array(DTRMSE_list)
DTRMSE=np.mean(DTRMSE_list)

# 평균 결과
df3['mean'] = df3.iloc[:, 0:repeatnum].mean(axis=1)

print('################################################')

###knn###
k = round(len(x_train) ** 0.5) #k=square root of the total number of samples
knn = KNeighborsRegressor(n_neighbors=k, weights="distance").fit(x_train, y_train)
predict_knn = knn.predict(x_test)
RMSE_knn = mean_squared_error(y_test_cut, predict_knn[5760:7200], squared=False)

print('################################################')

# 알고리즘별 train score과 RMSE 값

print(str(repeatnum)+'repeated-train score(adaboost):',adaboosttrain)
print(str(repeatnum)+'repeated-RMSE(adaboost) : ',adaboostRMSE)
print('Adaboost 직렬 모델의 RMSE 오차:', round(max(adaboostRMSE_list)-min(adaboostRMSE_list),3))
print()

print(str(repeatnum)+'repeated-train score(Decision Tree):', DTtrain)
print(str(repeatnum)+'repeated-RMSE(adaboost) : ',DTRMSE)
print('Decision Tree 직렬 모델의 RMSE 오차:', round(max(DTRMSE_list)-min(DTRMSE_list),3))
print()

print('train score(knn):',knn.score(x_train, y_train))
print('RMSE(knn) : ',RMSE_knn)
print()

###plot###
# test data: Thu~next Mon(y)

# Adaboost
plt.figure(figsize=(20, 10))
plt.plot(y_test, label='LPF real')
plt.plot(df2['mean'], label = 'adaboost')
ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(720))
plt.xlabel("DateTime")
plt.ylabel('Energy consumption per timeslot [kWh]')
plt.xticks(fontsize=10)
plt.legend()
plt.title('week' + str(w) + '- Adaboost series', size=20)
plt.show()

# Decision Tree
plt.figure(figsize=(20, 10))
plt.plot(y_test, label='LPF real')
plt.plot(df3['mean'], label = 'decision tree')
ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(720))
plt.xlabel("DateTime")
plt.ylabel('Energy consumption per timeslot [kWh]')
plt.xticks(fontsize=10)
plt.legend()
plt.title(' Decision Tree series', size=20)
plt.show()

#kNN
plt.figure(figsize=(20, 10))
plt.plot(y_test, label='LPF real')
plt.plot(predict_knn, label = 'knn')
ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(720))
plt.xlabel("DateTime")
plt.ylabel('Energy consumption per timeslot [kWh]')
plt.xticks(fontsize=10)
plt.legend()
plt.title(' kNN series', size=20)
plt.show()

#Plot All
plt.figure(figsize=(20, 10))
plt.plot(df3['mean'], label = 'Decision Tree prediction')
plt.plot(df2['mean'], label = 'Adaboost prediction')
plt.plot(predict_knn, label='kNN prediction')
plt.plot(y_test, label='LPF real')
ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(720))
plt.xlabel("index")
plt.ylabel('Energy consumption per timeslot [kWh]')
plt.xticks(fontsize=10)
plt.legend()
plt.title(' ALL series algorithms ('+str(repeatnum)+' repeated)', size=20)
plt.show()

