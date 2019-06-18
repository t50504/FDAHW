# FDA Term Project 
### Project name:登革熱 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Team: shallow learning
***
### 程式運行環境與相關函式庫
- 運行環境:使用python3.4以上版本，運行於Jupyter notebook ，按照Cell順序依序執行。

- 相關函式庫: 
```py
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from  sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO   
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,auc,roc_curve
from sklearn import svm,preprocessing
```
---
### 程式架構
- 資料前處理:醫療上的指數取第一天或最小天數時，並將散落在各sheet的病歷資料整合在一起
並將發病日和確診日整合成diag-onset值。
```py
sheet1=data.get('total').sort_values('chartno', ascending=True)
sheet1.set_index("chartno", inplace=True)

sheet2_finish=data.get('AST').sort_values('Day', ascending=True).drop_duplicates('chartno')
sheet2_finish=sheet2_finish.sort_values('chartno', ascending=True)
sheet2_finish.set_index("chartno", inplace=True)
sheet2_finish=sheet2_finish.rename(columns={"value":"AST_value"})

sheet3_finish=data.get('ALT').sort_values('Day', ascending=True).drop_duplicates('chartno')
sheet3_finish=sheet3_finish.sort_values('chartno', ascending=True)
sheet3_finish.set_index("chartno", inplace=True)
sheet3_finish=sheet3_finish.rename(columns={"value":"ALT_value"})

sheet4_finish=data.get('APTT').sort_values('Day', ascending=True).drop_duplicates('chartno')
sheet4_finish=sheet4_finish.sort_values('chartno', ascending=True)
sheet4_finish.set_index("chartno", inplace=True)
sheet4_finish=sheet4_finish.rename(columns={"value":"APTT_value"})

sheet5_finish=data.get('Platelet').sort_values('Day', ascending=True).drop_duplicates('chartno')
sheet5_finish=sheet5_finish.sort_values('chartno', ascending=True)
sheet5_finish.set_index("chartno", inplace=True)
sheet5_finish=sheet5_finish.rename(columns={"value":"Platelet_value"})

import time
import datetime

sheet2_finish=sheet2_finish.drop(['type', 'Day'],axis=1)
sheet3_finish=sheet3_finish.drop(['type', 'Day'],axis=1)
sheet4_finish=sheet4_finish.drop(['type', 'Day'],axis=1)
sheet5_finish=sheet5_finish.drop(['type', 'Day'],axis=1)

sheet=pd.concat([sheet1, sheet2_finish, sheet3_finish, sheet4_finish, sheet5_finish],axis=1)
for i in sheet.index[:]:
    result=datetime.datetime.strptime(sheet.loc[i, 'diag_date'], '%Y-%m-%d')-datetime.datetime.strptime(sheet.loc[i, 'onset_date'], '%Y-%m-%d')
    sheet.loc[i, 'diag-onset']=result.days

#丟棄無用
finaldata=sheet.drop(['onset_date', 'diag_date', 'death_date'],axis=1)
finaldata.head()
```
- 填補缺漏值:因樣本落差大，不適合單純用中位數或平均數填值，因此用隨機森林模型來填補缺漏值。
```py
from  sklearn.ensemble import RandomForestRegressor

def fillvalue(data,attrname):
#以資料完整的筆數建立預測空缺值模型
    tr=data[data[attrname].notnull()]
    tr_X=tr[['age','sex','is_hospitalization','diag-onset']] 
    tr_Y=tr[[attrname]] 
    tr_X=tr_X.astype(float)
    tr_Y=tr_Y.astype(float)

    te=data[data[attrname].isnull()]
    te_X=te[['age','sex','is_hospitalization','diag-onset']].astype(float) #設定輸入的X
    te_Y=te[attrname].astype(float) #欲填補的空值y
    
    fc=RandomForestRegressor()
    fc.fit(tr_X,tr_Y)
    pr=fc.predict(te_X)
    for i in range(len(pr)):
        te[attrname].values[i]=pr[i]
    return(te)

te_AST=fillvalue(finaldata,'AST_value')
final=pd.concat([finaldata, te_AST],axis=0)
final.reset_index(inplace=True)
final.drop_duplicates('index',keep='last',inplace=True)
final = final.rename(columns={'index': 'chartno'})
final_AST=final.reset_index(drop=True)

te_ALT=fillvalue(final_AST,'ALT_value')
final_ALT=pd.concat([final_AST, te_ALT],axis=0)
final_ALT.drop_duplicates('chartno',keep='last',inplace=True)
final_ALT=final_ALT.reset_index(drop=True)

te_APTT=fillvalue(final_ALT,'APTT_value')
final_APTT=pd.concat([final_ALT, te_APTT],axis=0)
final_APTT.drop_duplicates('chartno',keep='last',inplace=True)
final_APTT=final_APTT.reset_index(drop=True)

te_Platelet=fillvalue(final_APTT,'Platelet_value')
final_Platelet=pd.concat([final_APTT, te_Platelet],axis=0)
final_Platelet.drop_duplicates('chartno',keep='last',inplace=True)
final_Platelet=final_Platelet.reset_index(drop=True)

finaldata=final_Platelet.sort_values('chartno', ascending=True)
finaldata.set_index("chartno", inplace=True)
finaldata.head()
```
 資料前處理完畢，接下來進入資料分析階段:
- 首先應用熱圖探討標籤值(是否死亡)與其餘特徵的相關性
```py
plt.figure(figsize=(8, 8))
sns.heatmap(finaldata.corr(), annot=True).set_title('Correlation')
```
- 接著從相關性較大的特徵深入探討，將資料視覺化，藉此觀察趨勢與規律:
分別是  性別與存活；住院與否與存活；年齡與住院與存活；登革熱好發時期趨勢； 4個醫療指數與存活。
```py
#性別與存活關係圖
plt.figure(figsize=(8, 6))
labels = 'Female','Male'
FS=list(finaldata.loc[finaldata['Fatal']==0,'sex']).count(0)
FD=list(finaldata.loc[finaldata['Fatal']==1,'sex']).count(0)
MS=list(finaldata.loc[finaldata['Fatal']==0,'sex']).count(1)
MD=list(finaldata.loc[finaldata['Fatal']==1,'sex']).count(1)
size1 = [FS,MS]
size2 = [FD,MD]
plt.subplot(1,2,1)
plt.pie(size1, labels = labels, autopct='%1.1f%%',explode=(0,0.1))
plt.title("Survive")
plt.subplot(1,2,2)
plt.pie(size2, labels = labels, autopct='%1.1f%%',explode=(0,0.1))
plt.title("Die")
plt.show

#住院與否與存活關係圖
plt.figure(figsize=(8, 6))
labels1 = 'Survive','Die'
NHS=list(finaldata.loc[finaldata['Fatal']==0,'is_hospitalization']).count(0)
NHD=list(finaldata.loc[finaldata['Fatal']==1,'is_hospitalization']).count(0)
HS=list(finaldata.loc[finaldata['Fatal']==0,'is_hospitalization']).count(1)
HD=list(finaldata.loc[finaldata['Fatal']==1,'is_hospitalization']).count(1)
size1 = [NHS,NHD]
size2 = [HS,HD]
plt.subplot(1,2,1)
plt.pie(size1, labels = labels1, autopct='%1.1f%%',explode=(0,0.2))
plt.title("No Hospitalization")
plt.subplot(1,2,2)
plt.pie(size2, labels = labels1, autopct='%1.1f%%',explode=(0,0.2))
plt.title("Hospitalization")
plt.show

#年齡與存活關係圖
plt.figure(figsize=(16, 4))
plt.subplot(1,2,1)
plt.bar(finaldata["age"].groupby([finaldata.loc[finaldata['Fatal']==1,"age"]]).count().index, finaldata["age"].groupby([finaldata.loc[finaldata['Fatal']==1,"age"]]).count())
plt.plot(finaldata["age"].groupby([finaldata.loc[finaldata['Fatal']==1,"age"]]).count().index, finaldata["age"].groupby([finaldata.loc[finaldata['Fatal']==1,"age"]]).count(),c='red',ls='-',lw=2)
plt.xlabel("Age") 
plt.ylabel("Number of Death") 

plt.subplot(1,2,2)
plt.bar(finaldata["age"].groupby([finaldata.loc[finaldata['is_hospitalization']==1,"age"]]).count().index, finaldata["age"].groupby([finaldata.loc[finaldata['is_hospitalization']==1,"age"]]).count())
plt.plot(finaldata["age"].groupby([finaldata.loc[finaldata['is_hospitalization']==1,"age"]]).count().index, finaldata["age"].groupby([finaldata.loc[finaldata['is_hospitalization']==1,"age"]]).count(),c='red',ls='-',lw=2)
plt.xlabel("Age") 
plt.ylabel("Number of Hospitalization") 

#登革熱好發時期
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
onset_month=np.zeros(12)
diag_month=np.zeros(12)
death_month=np.zeros(12)
for i in sheet1.index[:]:
    onset_month[datetime.datetime.strptime(sheet1.loc[i, 'onset_date'], '%Y-%m-%d').month-1]=onset_month[datetime.datetime.strptime(sheet1.loc[i, 'onset_date'], '%Y-%m-%d').month-1]+1
    diag_month[datetime.datetime.strptime(sheet1.loc[i, 'diag_date'], '%Y-%m-%d').month-1]=diag_month[datetime.datetime.strptime(sheet1.loc[i, 'diag_date'], '%Y-%m-%d').month-1]+1
    if sheet1.loc[i,'Fatal']==1:
        death_month[datetime.datetime.strptime(sheet1.loc[i, 'death_date'], '%Y-%m-%d').month-1]=death_month[datetime.datetime.strptime(sheet1.loc[i, 'death_date'], '%Y-%m-%d').month-1]+1
plt.bar(np.arange(1,13),onset_month,0.35,label='onset')
plt.bar(np.arange(1,13)+0.35,diag_month,0.35,label='diag')
plt.xticks(np.arange(1,13)+0.175,np.arange(1,13))
plt.legend()
plt.xlabel("Month") 
plt.ylabel("Number") 

plt.subplot(1,2,2)
plt.bar(np.arange(1,13),death_month,0.35,label='death')
plt.xticks(np.arange(1,13),np.arange(1,13))
plt.xlabel("Month") 
plt.ylabel("Number") 
plt.legend()

#四個醫療指數與存活關聯圖
plt.figure(figsize=(16, 6))
plt.subplot(1,2,1)
X1=finaldata.loc[finaldata['Fatal']==0,'AST_value']
Y1=finaldata.loc[finaldata['Fatal']==0,'ALT_value']
c1="green"
X2=finaldata.loc[finaldata['Fatal']!=0,'AST_value']
Y2=finaldata.loc[finaldata['Fatal']!=0,'ALT_value']
c2="red"
plt.scatter(X1, Y1, marker='o', c=c1, alpha=.3, label="Survive")
plt.scatter(X2, Y2, marker='x', c=c2, alpha=1, label="Die")
plt.xlabel('AST_value')
plt.ylabel('ALT_value')
plt.legend(loc="upper left")

plt.subplot(1,2,2)
X1=finaldata.loc[finaldata['Fatal']==0,'APTT_value']
Y1=finaldata.loc[finaldata['Fatal']==0,'Platelet_value']
c1="green"
X2=finaldata.loc[finaldata['Fatal']!=0,'APTT_value']
Y2=finaldata.loc[finaldata['Fatal']!=0,'Platelet_value']
c2="red"
plt.scatter(X1, Y1, marker='o', c=c1, alpha=.3, label="Survive")
plt.scatter(X2, Y2, marker='x', c=c2, alpha=1, label="Die")
plt.xlabel('APTT_value')
plt.ylabel('Platelet_value')
plt.legend(loc="upper left")

plt.show()
```
資料分析後可得到一些趨勢報告，將詳細於文檔中探討。

- 接著，為了預測病患存活與否，開始將上述處理完成的資料拿來訓練機器學習模型，因有許多特徵數值，為了初步看出成果，故我們先應用Decision Tree Model ，也將其當作Baseline model。
```py
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(finaldata.loc[:, finaldata.columns!='Fatal'], finaldata.loc[:, 'Fatal'], test_size=0.3, random_state=1)
dtree=DecisionTreeClassifier(max_depth=4)
dtree.fit(X_train,y_train)

dot_data = StringIO()
export_graphviz(dtree, 
                out_file=dot_data,  
                filled=True, 
                feature_names=list(X_train),
                class_names=['die','survive'],
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png("tree.png")

y_predict = dtree.predict(X_test)
print("\n準確度 : ", accuracy_score(y_test, y_predict))
pd.DataFrame([dtree.feature_importances_],index=None, columns = X_train.columns)
```
預測準確度達到99%，試著捨棄一些相關性低的特徵，卻仍舊保持著99%的準確度。看似是不錯的成果。
使用混淆矩陣及ROC Curve來評估看看結果是否那麼好。
```py
C=confusion_matrix(y_test, y_predict)
ax = plt.axes()
sns.heatmap(C, annot=True,ax=ax)
ax.set_title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

FPR=C[0][1]/(C[0][1]+C[1][1])*100
TPR=C[0][0]/(C[0][0]+C[1][0])*100
P=C[0][0]/(C[0][0]+C[0][1])*100
R=C[0][0]/(C[0][0]+C[1][0])*100
F1=2*P*R/(P+R)

scores=[[P,R,FPR,TPR,F1]]
subject = ['Precision', 'Recall', 'FPR', 'TPR', 'F1']
df = pd.DataFrame(scores,index=None, columns = subject)
df

y_score = dtree.fit(X_train1, y_train).predict(X_test1)

fpr,tpr,threshold = roc_curve(y_test, y_score) #計算FPR和TPR
roc_auc = auc(fpr,tpr) #計算auc的值
 
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)\nAUC = %0.2f' % (roc_auc,roc_auc)) #FPR為X軸，TPR為Y軸
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="upper left")
plt.show()
```
FPR竟然達到100%，且ROC Curve也是等於無效的50%斜直線，看來實際效果並不如預期。
- 接著嘗試看看SVM Model
```py
classifier = svm.SVC(1, kernel='rbf',gamma='auto') 
classifier.fit(X_train1,y_train)
y_predict2=classifier.predict(X_test1)
print("準確度 : ", accuracy_score(y_test, y_predict2))

C=confusion_matrix(y_test, y_predict2)
ax = plt.axes()
sns.heatmap(C, annot=True,ax=ax)
ax.set_title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

FPR=C[0][1]/(C[0][1]+C[1][1])*100
TPR=C[0][0]/(C[0][0]+C[1][0])*100
P=C[0][0]/(C[0][0]+C[0][1])*100
R=C[0][0]/(C[0][0]+C[1][0])*100
F1=2*P*R/(P+R)

scores=[[P,R,FPR,TPR,F1]]
subject = ['Precision', 'Recall', 'FPR', 'TPR', 'F1']
df = pd.DataFrame(scores,index=None, columns = subject)
df

y_score = classifier.fit(X_train1, y_train).decision_function(X_test1)

fpr,tpr,threshold = roc_curve(y_test, y_score) #計算FPR和TPR
roc_auc = auc(fpr,tpr) #計算auc的值
 
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)\nAUC = %0.2f' % (roc_auc,roc_auc)) #FPR為X軸，TPR為Y軸
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="upper left")
plt.show()
```
在混淆矩陣跟ROC曲線中都有較佳的表現。或許此一模型將是堪用的預測模型。
