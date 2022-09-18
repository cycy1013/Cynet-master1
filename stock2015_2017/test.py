import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("stock2015_2017/models/3modeluse85models#ALL#close.csv")
#print("df:\n",df)
df_TP=df[(df['actual_event']==1) & (df['predictions']==1) ]
df_FN=df[(df['actual_event']==1) & (df['predictions']==0) ]
df_FP=df[(df['actual_event']==0) & (df['predictions']==1) ]
df_TN=df[(df['actual_event']==0) & (df['predictions']==0) ]
print("实际的positive数是:",len(df[df['actual_event']==1]))
print("实际的negative数是:",len(df[df['actual_event']==0]))
print("len(df_TP):",len(df_TP))
print("len(df_FP):",len(df_FP))
print("len(df_TN):",len(df_TN))
print("len(df_FN):tp:",len(df_FN))
print("全部数据 tpr:",len(df_TP)/(len(df_TP)+len(df_FN)))
print("全部数据 fpr:",len(df_FP)/(len(df_FP)+len(df_TN)))

#print(len(df1)/len(df))
#print(df1['day'].unique())
df_near=df[693:703]
#print("df_p:\n",df_near)
df_TP=df_near[(df_near['actual_event']==1) & (df_near['predictions']==1) ]
df_FN=df_near[(df_near['actual_event']==1) & (df_near['predictions']==0) ]
df_FP=df_near[(df_near['actual_event']==0) & (df_near['predictions']==1) ]
df_TN=df_near[(df_near['actual_event']==0) & (df_near['predictions']==0) ]
ax = sns.violinplot(x=df['actual_event'], y=df['positive_event'], data=df, cut=0,showmeidans=True)  # 小提琴图，看分布
print('事件时,预测发生的概率中位数:',df[df['actual_event']==1]['positive_event'].median())
print('事件不发生时,预测发生的概率中位数:',df[df['actual_event']==0]['positive_event'].median())
plt.show()

print("len(df_TP):",len(df_TP))
print("len(df_FP):",len(df_FP))
print("len(df_TN):",len(df_TN))
print("len(df_FN):tp:",len(df_FN))
print("最后10天tpr:",len(df_TP)/(len(df_TP)+len(df_FN)))
print("最后10天fpr:",len(df_FP)/(len(df_FP)+len(df_TN)))

df_p=df[713:]
print("df_p:\n",df_p)
df_TP=df_p[(df_p['actual_event']==1) & (df_p['predictions']==1) ]
df_FN=df_p[(df_p['actual_event']==1) & (df_p['predictions']==0) ]
df_FP=df_p[(df_p['actual_event']==0) & (df_p['predictions']==1) ]
df_TN=df_p[(df_p['actual_event']==0) & (df_p['predictions']==0) ]
print("len(df_TP):",len(df_TP))
print("len(df_FP):",len(df_FP))
print("len(df_TN):",len(df_TN))
print("len(df_FN):tp:",len(df_FN))
print("预测5天tpr:",len(df_TP)/(len(df_TP)+len(df_FN)))
print("预测5天fpr:",len(df_FP)/(len(df_FP)+len(df_TN)))
