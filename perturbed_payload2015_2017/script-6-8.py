import cynet.cynet as cn
import numpy as np
import sys
sys.path.append("/usr/local/lib64/")
print("Path:\n",sys.path)
import pickle
EPS = 10     #200     EPS=10，会把这个城市分成10*10=100个分区
STOREFILE='crime.p'
CSVFILE='Crimes_-_2001_to_Present.csv'


# #Section 3: script 6: 生成预测log文件
##针对一个模型文件(针对一个target(经纬度+类型))，分别处理它的每一类src variable,并生成对应的log预测文件(对一个目标块、目标变量，多个源块的一个源变量的多个delta时延的预测),最终会生成多个log文件(每个对应一类src变量)
# import cynet.cynet as cn
# import yaml
# import glob
# with open('perturbed_payload2015_2017/config.yaml','r') as fh:
#     settings_dict = yaml.safe_load(fh)
# model_nums = settings_dict['model_nums']#the number of model want to use old:85,new:31
# MODEL_GLOB = settings_dict['MODEL_GLOB'] #*model.json
# horizon = settings_dict['horizons'][0] #[7]
# DATA_PATH = settings_dict['DATA_PATH'] #./split_burg_10p/2015-01-01_2018-12-31,#这个时段的多个块多个犯罪类型的一行犯罪记录
# RUNLEN = settings_dict['RUNLEN'] #365*4=1460
# RESPATH = settings_dict['RESPATH'] #*model*res
# FLEX_TAIL_LEN = settings_dict['FLEX_TAIL_LEN'] #365
# # #glob.glob()查找符合特定规则的文件路径名
# VARNAME=list(set([i.split('#')[-1] for i in glob.glob(DATA_PATH+"*")]))+['ALL']
# print("VARNAME:\n",VARNAME)#['BURGLARY-THEFT-MOTOR_VEHICLE_THEFT', 'VAR', 'HOMICIDE-ASSAULT-BATTERY', 'ALL']
# #cn.run_pipeline(MODEL_GLOB,model_nums, horizon, DATA_PATH, RUNLEN, VARNAME, RESPATH,FLEX_TAIL_LEN=FLEX_TAIL_LEN,cores=4,gamma=True)
#cn.run_pipeline(MODEL_GLOB,model_nums, horizon, DATA_PATH, RUNLEN, VARNAME, RESPATH,FLEX_TAIL_LEN=FLEX_TAIL_LEN,cores=1,gamma=True) #gamma:是否按gamma排序

# #Section 3:script 7: Plotting statistics绘制预测统计图
# import cynet.cynet as cn
# VARNAMES=['BURGLARY-THEFT-MOTOR_VEHICLE_THEFT','HOMICIDE-ASSAULT-BATTERY','VAR']
# #res_all.csv：针对所有模型(地块+犯罪类型)的所有源类型的预测统计值
# #所有目标块的所有目标变量的统计值(tpr,auc,fpr)的分布情况
# cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2','vartgt'],varname='auc',VARNAMES=VARNAMES)
# cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2','vartgt'],varname='tpr',VARNAMES=VARNAMES)
# cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2','vartgt'],varname='fpr',VARNAMES=VARNAMES)
# #所有目标块的统计值(tpr,auc,fpr)的分布情况
# cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2'],varname='tpr',VARNAMES=VARNAMES)
# cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2'],varname='auc',VARNAMES=VARNAMES)
# cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2'],varname='fpr',VARNAMES=VARNAMES)

#section 4: script 8:根据设定的tpr,得到阈值,然后判断事件是否发生,把每个预测log文件扩充成预测csv文件
import cynet.cynet as cn
import yaml
with open('perturbed_payload2015_2017/config.yaml','r') as fh:
    settings_dict = yaml.safe_load(fh)
FLEX_TAIL_LEN = settings_dict['FLEX_TAIL_LEN']
#cn.flexroc_only_parallel('perturbed_payload2015_2017/models/*.log',tpr_threshold=0.85,fpr_threshold=None,FLEX_TAIL_LEN=FLEX_TAIL_LEN, cores=4)
cn.flexroc_only_parallel('perturbed_payload2015_2017/models/*.log',tpr_threshold=0.85,fpr_threshold=None,FLEX_TAIL_LEN=FLEX_TAIL_LEN, cores=1)

# #Section 5 script 9:合并各预测csv文件 得到各个目标块的各目标变量的指定源变量的预测,方便绘制热力图(有各目标块的信息)
# import cynet.cynet as cn
# mapper=cn.mapped_events('perturbed_payload2015_2017/models/*85models#ALL#*.csv')
# mapper.concat_dataframes('perturbed_payload2015_2017/models/85modelsALL.csv')

# #Section 5 Script 10:绘制热力图
# import viscynet.viscynet as viz
# import numpy as np
# import pandas as pd
# import yaml
# with open('perturbed_payload2015_2017/config2.yaml','r') as fh:
#     settings_dict = yaml.safe_load(fh)
# source = settings_dict['source']
# types = settings_dict['types']
# grace = settings_dict['grace']
# EPS = settings_dict['EPS']
# lat_min = settings_dict['lat_min']
# lat_max = settings_dict['lat_max']
# lon_min = settings_dict['lon_min']
# lon_max = settings_dict['lon_max']
# day = settings_dict['day']
# csv = settings_dict['predictions_csv']
# shapefiles = settings_dict['shapefiles']
# radius = settings_dict['radius']
# detail = settings_dict['detail']
# min_intensity = settings_dict['min_intensity']
# df = pd.read_csv(csv)
# dt,fp,fn,tp,df_gnd_augmented,lon_mesh,lat_mesh,intensity = \
# viz.get_prediction(df,day,lat_min,
# lat_max,lon_min,lon_max,source,
# types,startdate="12/31/2017",offset=1095,
# radius=radius,detail=detail,
# Z=min_intensity,SINGLE=False)
# #print("lat_mesh:\n",lat_mesh,"lon_mesh:\n",lon_mesh)
# viz.getFigure(day,dt,fp,fn,tp,df_gnd_augmented,lon_mesh,lat_mesh,\
#               intensity,fname=shapefiles,cmap='terrain', save=True,PREFIX='Burglary')

## Section 6:script 11:扰动测试数据 
##SBURGLARY-THEFT-MOTOR_VEHICLE_THEFT 增加10%
#cp split/2015-01-01_2018-12-31*VAR  split_burg_10p/
#cp split/2015-01-01_2018-12-31*HOMICIDE-ASSAULT-BATTERY  split_burg_10p/
#cp payload2015_2017/models/*model.json  perturbed_payload2015_2017/models/*model.json
# import cynet.cynet as cn
# cn.alter_splitfiles('split/2015*BURGLARY-THEFT-MOTOR_VEHICLE_THEFT','split_burg_10p/', theta=0.1)
