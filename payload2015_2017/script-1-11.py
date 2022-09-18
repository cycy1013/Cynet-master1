import cynet.cynet as cn
import numpy as np
import sys
#sys.path.append("/usr/local/lib64/")
#print("Path:\n",sys.path)
import pickle
EPS = 10     #200     EPS=10，会把这个城市分成9*9=81个分区
STOREFILE='crime.p'
CSVFILE='Crimes_-_2001_to_Present.csv' # Crimes_-_2022.csv ;Crimes_-_2001_to_Present.csv

# #section 1 数据预处理:Script 1:
# #creates tiles.txt, crime.p, and CRIME-BURGLARY-THEFT-MOTOR_VEHICLE_THEFT.csv.
# grid={'Latitude':np.around(np.linspace(41.5,42.05,EPS),decimals=5),
# 'Longitude':np.around(np.linspace(-87.9,-87.2,EPS),decimals=5),
# 'Eps':EPS}
# tiles=list([[grid['Latitude'][i],grid['Latitude'][i+1],grid['Longitude'][j], grid['Longitude'][j+1]]
#     for i in np.arange(len(grid['Latitude'])-1)
#         for j in np.arange(len(grid['Longitude'])-1)])
# #print(np.array(tiles).shape) # 9*9=81,4
# #生成crime.p,CRIME-BURGLARY-THEFT-MOTOR_VEHICLE_THEFT.csv,tiles.txt这三个文件
# S0=cn.spatioTemporal(log_file=CSVFILE,
#     log_store=STOREFILE,#放罪记录写到这个文件，以后就不用从CSVFILE文件中读取了
#     types=[['BURGLARY','THEFT','MOTOR VEHICLE THEFT']],#从原始犯罪文件中过滤出这几个类型
#     value_limits=None,
#     grid=tiles, #从原始犯罪文件中过滤出这个空间范围的
#     init_date='2015-01-01', #2001-01-01 从原始犯罪文件中过滤出这个时段的
#     end_date ='2018-12-31', # '2022-12-31' 2018-12-31
#     freq='D',
#     EVENT='Primary Type',#列名,类型过滤所针对的列名
#     threshold=0.05)
# S0.fit(csvPREF='CRIME-') #筛选出合适的数据(类型,空间,时间,阈值)
# tiles=S0.getGrid() #由于把犯罪率过低的空间块过滤掉了，所以现在的空间块与最初设置的tiles有所不同,返回的是现在的空间块
# with open("tiles.txt", "wb") as tiles_pickle:
#     pickle.dump(tiles,tiles_pickle)

# #section 1 数据预处理:Script 2:生成CRIME-HOMICIDE-ASSAULT-BATTERY.csv
# STOREFILE='crime.p'
# with open("tiles.txt", "rb") as tiles_pickle:
#     tiles = pickle.load(tiles_pickle)
# S01=cn.spatioTemporal(log_store=STOREFILE,# 与script 1不同 ,没有log_file
#     types=[['HOMICIDE','ASSAULT','BATTERY']],
#     value_limits=None,
#     grid=tiles,
#     init_date='2015-01-01', #"2001-01-01"    4*365=1460,有闰年，所以少了一天(最后一天)
#     end_date='2018-12-31',   #'2018-12-31','2022-12-31'
#     freq='D',threshold=0.05)#  与script1不同,没有 EVENT，但默认就是'Primary Type'
# S01.fit(csvPREF='CRIME-')

# # section 1 数据预处理: Script 3:#生成ARREST.csv
# with open("tiles.txt", "rb") as tiles_pickle:
#     tiles = pickle.load(tiles_pickle)
# S2=cn.spatioTemporal(log_store=STOREFILE,# 与script 1不同 ,没有log_file
#     types=None,
#     value_limits=[0,1],# 与script 1不同,None; value_limits与types只能设置一个，不能同时设置 [0,1] 应该设成[0.5,1] modi by cy???
#     grid=tiles,
#     init_date='2015-01-01',
#     end_date='2018-12-31',#'2018-12-31','2022-12-31'
#     #local_func=sum, #added by cy
#     freq='D',
#     EVENT='Arrest',# 与script 1不同 ,EVENT默认是'Primary Type';Arrest列是bool型（EVENT:类型过滤所针对的列名）
#     threshold=0.05)  #0.05
# S2.fit(csvPREF='ARREST')

# section 1 数据预处理:Script 4:
#生成训练用的数据 triplet/,用于预测的数据 split/   少掉的一天又怎么多出来的??????
# 训练数据(1) triplet/CRIME-_2015-01-01_2017-12-31.columns
# 训练数据(2) triplet/CRIME-_2015-01-01_2017-12-31.coords
# 训练数据(3) triplet/CRIME-_2015-01-01_2017-12-31.csv
import cynet.cynet as cn
CSVfile = ['ARREST.csv','CRIME-BURGLARY-THEFT-MOTOR_VEHICLE_THEFT.csv','CRIME-HOMICIDE-ASSAULT-BATTERY.csv']
begin = '2015-01-01'
end = '2017-12-31'
extended_end = '2018-12-31'
name = 'triplet/' + 'CRIME-'+'_' + begin + '_' + end
#Generates desired triplets.训练用的数据(3年的)
cn.readTS(CSVfile,csvNAME=name,BEG=begin,END=end)
#Generates files which contains in sample and out of sample data.
#split文件内容只有一行，1460列(这个时间段的每天),文件名是:时间段+经纬度+犯罪类型，值是这一天的这一地块内的这一类别的犯罪计数
cn.splitTS(CSVfile, BEG = begin, END = extended_end, dirname = './split', prefix = begin + '_' + extended_end)

# #Section 2 生成模型:Script 5:(需linux环境下运行)  模型的含义?
# #比如:./payload2015_2017/models/10model.json
# #针对一个target空间块(准确地说是：经纬度+犯罪类型) 每个src的影响,每个 Δ天
# import cynet.cynet as cn
# import yaml
# with open('payload2015_2017/config.yaml','r',encoding="utf-8") as fh:
#     settings_dict = yaml.safe_load(fh)
# print("settings_dict:\n",settings_dict)
# TS_PATH=settings_dict['TS_PATH'] #'./triplet/CRIME-_2015-01-01_2017-12-31.csv'
# NAME_PATH=settings_dict['NAME_PATH']#'./triplet/CRIME-_2015-01-01_2017-12-31.coords'
# LOG_PATH=settings_dict['LOG_PATH'] #'log.txt'  最后形成10log.txt,11log.txt...??
# FILEPATH=settings_dict['FILEPATH'] #'./payload2015_2017/models/'
# END=settings_dict['END'] #60
# BEG=settings_dict['BEG'] #0
# NUM=settings_dict['NUM'] #2 ?? 并行运行的进程个数
# PARTITION=settings_dict['PARTITION'] #'PARTITION': [0.5]??
# XgenESeSS=settings_dict['XgenESeSS']
# RUN_LOCAL=settings_dict['RUN_LOCAL']#本地运行还是在集群中运行
# XG = cn.xgModels(TS_PATH,NAME_PATH, LOG_PATH,FILEPATH, BEG, END, NUM, PARTITION, XgenESeSS,RUN_LOCAL)
# XG.run(workers=1) #workers=4
# # #读出json文件并格式化显示出来
# # import json
# # with open("./payload2015_2017/models/10model.json",encoding="utf-8") as f:
# #     result=json.load(f)
# #     #print("result:\n",result,"\n",type(result))
# #     result_str=json.dumps(result,indent=2)
# #     print("result_str:\n",result_str)

# #Section 3: script 6: 生成预测log文件
# import cynet.cynet as cn
# import yaml
# import glob
# with open('payload2015_2017/config.yaml','r',encoding="utf-8") as fh:
#     settings_dict = yaml.safe_load(fh)
# model_nums = settings_dict['model_nums'] # the number of model want to use old:85,new:31
# MODEL_GLOB = settings_dict['MODEL_GLOB'] #*model.json
# horizon = settings_dict['horizons'][0] #[7]
# DATA_PATH = settings_dict['DATA_PATH'] #./split/2015-01-01_2018-12-31
# RUNLEN = settings_dict['RUNLEN'] #365*4=1460
# RESPATH = settings_dict['RESPATH'] #*model*res
# FLEX_TAIL_LEN = settings_dict['FLEX_TAIL_LEN'] #365
# #glob.glob()查找符合特定规则的文件路径名
# VARNAME=list(set([i.split('#')[-1] for i in glob.glob(DATA_PATH+"*")]))+['ALL']
# print("VARNAME:\n",VARNAME) #['BURGLARY-THEFT-MOTOR_VEHICLE_THEFT', 'VAR', 'HOMICIDE-ASSAULT-BATTERY', 'ALL']
# #cn.run_pipeline(MODEL_GLOB,model_nums, horizon, DATA_PATH, RUNLEN, VARNAME, RESPATH,FLEX_TAIL_LEN=FLEX_TAIL_LEN,cores=4,gamma=True)
# cn.run_pipeline(MODEL_GLOB,model_nums, horizon, DATA_PATH, RUNLEN, VARNAME, RESPATH,FLEX_TAIL_LEN=FLEX_TAIL_LEN,cores=1,gamma=True) #gamma:是否按gamma排序

# import json
# with open("./payload2015_2017/models/2model_sel_21adc854-47c1-4890-a02a-9c8ed831b5bb.json",encoding="utf-8") as f:
#     result=json.load(f)
#     #print("result:\n",result,"\n",type(result))
#     result_str=json.dumps(result,indent=2)
#     print("result_str:\n",result_str)


# #Section 3:script 7: Plotting statistics绘制预测统计图
# import cynet.cynet as cn
# VARNAMES=['BURGLARY-THEFT-MOTOR_VEHICLE_THEFT','HOMICIDE-ASSAULT-BATTERY','VAR']
# cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2','vartgt'],varname='auc',VARNAMES=VARNAMES)
# cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2','vartgt'],varname='tpr',VARNAMES=VARNAMES)
# cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2','vartgt'],varname='fpr',VARNAMES=VARNAMES)
# cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2'],varname='tpr',VARNAMES=VARNAMES)
# cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2'],varname='auc',VARNAMES=VARNAMES)
# cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2'],varname='fpr',VARNAMES=VARNAMES)

# #section 4: script 8:根据设定的tpr,得到阈值,然后判断事件是否发生,把每个预测log文件扩充成预测csv文件
# import cynet.cynet as cn
# import yaml
# with open('payload2015_2017/config.yaml',encoding="utf-8") as fh:
#     settings_dict = yaml.safe_load(fh)
# FLEX_TAIL_LEN = settings_dict['FLEX_TAIL_LEN']
# #cn.flexroc_only_parallel('payload2015_2017/models/*.log',tpr_threshold=0.85,fpr_threshold=None,FLEX_TAIL_LEN=FLEX_TAIL_LEN, cores=4)
# #每个模型(目标块+犯罪类型)的每类源变量都有一个log预测文件
# cn.flexroc_only_parallel('payload2015_2017/models/*.log',tpr_threshold=0.85,fpr_threshold=None,FLEX_TAIL_LEN=FLEX_TAIL_LEN, cores=1)

# #Section 5 script 9:合并各预测csv文件 得到各个目标块的各目标变量的指定源变量的预测,方便绘制热力图(有各目标块的信息)
# import cynet.cynet as cn
# #mapper=cn.mapped_events('payload2015_2017/models/*85models#ALL#*.csv')
# mapper=cn.mapped_events('payload2015_2017/models/*31models#ALL#*.csv')
# #mapper.concat_dataframes('payload2015_2017/models/85modelsALL.csv')
# mapper.concat_dataframes('payload2015_2017/models/31modelsALL.csv')

# #Section 5 Script 10:绘制热力图
# import viscynet.viscynet as viz
# import numpy as np
# import pandas as pd
# import yaml
# with open('payload2015_2017/config2.yaml','r') as fh:
#     settings_dict = yaml.safe_load(fh)
# source = settings_dict['source'] #源变量 'ALL'
# types = settings_dict['types'] #目标变量 ["BURGLARY-THEFT-MOTOR_VEHICLE_THEFT"]
# grace = settings_dict['grace'] #允许正负几天的误差(这个范围内认为预测正确)
# EPS = settings_dict['EPS'] #分成 (EPS-1)*(EPS-1)个网格
# lat_min = settings_dict['lat_min']
# lat_max = settings_dict['lat_max']
# lon_min = settings_dict['lon_min']
# lon_max = settings_dict['lon_max']
# day = settings_dict['day'] #1415,代表2018-11-16这一天(20150101-20171231 1095天)
# csv = settings_dict['predictions_csv'] #85modelsALL.csv:各个目标块的各目标变量的ALL源变量的预测
# shapefiles = settings_dict['shapefiles'] #城市的形状文件
# radius = settings_dict['radius'] #涉及到画出来的点的半径
# detail = settings_dict['detail'] #经纬度的跳次(np.arange中的step)
# min_intensity = settings_dict['min_intensity']
# df = pd.read_csv(csv)
# #intensity：在代表目标块附近的经纬度的网格上，添加了高斯分布的随机值(网格上其他点的值为0) (696, 546)
# #df_gnd_augmented：当天实际发生了，前一天又预测发生了；当天实际发生了
# dt,fp,fn,tp,df_gnd_augmented,lon_mesh,lat_mesh,intensity = \
#     viz.get_prediction(df,day,lat_min,
#     lat_max,lon_min,lon_max,source,
#     types,startdate="12/31/2017",offset=1095,
#     radius=radius,detail=detail,
#     Z=min_intensity,SINGLE=False)
# #print("lat_mesh:\n",lat_mesh,"lon_mesh:\n",lon_mesh)
# viz.getFigure(day,dt,fp,fn,tp,df_gnd_augmented,lon_mesh,lat_mesh,\
#               intensity,fname=shapefiles,cmap='terrain', save=True,PREFIX='Burglary')

# # Section 6:script 11:扰动测试数据
# #SBURGLARY-THEFT-MOTOR_VEHICLE_THEFT 增加10%
# # cp split/2015-01-01_2018-12-31*VAR  split_burg_10p/    #这个时段的多个块VAR类型的记录
# # cp split/2015-01-01_2018-12-31*HOMICIDE-ASSAULT-BATTERY  split_burg_10p/   #这个时段的HOMICIDE类型的记录
# #上面的两类数据不动(原封不动拷过来),BURGLARY的数据增加10%
# # cp payload2015_2017/models/*model.json  perturbed_payload2015_2017/models/*model.json
# import cynet.cynet as cn
# cn.alter_splitfiles('split/2015*BURGLARY-THEFT-MOTOR_VEHICLE_THEFT','split_burg_10p/', theta=0.1)
