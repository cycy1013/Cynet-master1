from tqdm import tqdm
# for i in tqdm(['a','b','c']):
#      print(i)
# import pandas as pd
# import numpy as np
# ps1=pd.Series([1,2,3])
# print('ps1:\n',ps1[[0]])
# print(ps1.size)

# df=pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]],columns=['a','b','c'])
# print("df:\n",df)
# print("df[['a']]\n",df[['a']],type(df[['a']]))
# print("df['a']:\n",df['a'],type(df['a']))

# aa=pd.concat([pd.DataFrame(np.arange(6)),pd.DataFrame(np.arange(6))])
# print("aa:\n",aa)

# import sys
# import os
# modules1 = sys.modules.copy()
# for key, value in modules1.items():
#     print('"{}": "{}"'.format(key, value))
#
# import flask
# modules2 = sys.modules.copy()
# # print('首次导入logging模块后在sys.modules中加入的模块名：',(modules2['flask'].__file__))
# import cynet
# print(os.path.dirname(sys.modules['cynet'].__file__))
# print(sys.modules['cynet'].__file__)

import multiprocessing


def function_square(data):
     result = data * data
     return result


# if __name__ == "__main__":
#      inputs = list(range(100))
#      print("inputs:\n",inputs)
#      pool = multiprocessing.Pool(processes=4)
#      pool_outputs = pool.map(function_square, inputs)
#      pool.close()
#      pool.join()
#      print("pool: ", pool_outputs)

from mpi4py import MPI


def mpi_test(rank):
     print("I am rank %s" % rank)

#D:\高博软件技术学院\新区教学\大数据\强化学习\百度强化学习-培训\Cynet-master>mpiexec -n 5 python test.py
# if __name__ == "__main__":
#      comm = MPI.COMM_WORLD  #通讯
#      rank = comm.Get_rank()
#      mpi_test(rank)
#      print("Hello world from process", rank)

# from mpi4py import MPI
#
# comm = MPI.COMM_WORLD
# rank = comm.rank
# print("My rank is :", rank)
#
# if rank == 0: #0=>4
#      data = 10000000
#      destination_process = 4
#      comm.send(data, dest=destination_process)
#      print("sending data %s to process %d" % (data, destination_process))
#
# if rank == 1:  #1=>8
#      destination_process = 8
#      data = "hello,I am rank 1"
#
#      comm.send(data, dest=destination_process)
#      print("sending data %s to process %d" % (data, destination_process))
#
# if rank == 4:#0=>4
#      data = comm.recv(source=0)
#      print("data received is = %s" % data)
#
# if rank == 8:#1=>8
#      data1 = comm.recv(source=1)
#      print("data received is = %s" % data1)

# #死锁的情况:
# from mpi4py import MPI
#
# comm = MPI.COMM_WORLD
# rank = comm.rank
# print("my rank is :", rank)
#
# if rank == 1: #1 <=5 ; 1=》5
#      data_send = "a"
#      destination_process = 5
#      source_process = 5
#      # comm.send(data_send, dest=destination_process)
#      # data_received = comm.recv(source=source_process)
#      data_received=comm.sendrecv(data_send,dest=destination_process,source=source_process)
#
#      print("sending data %s to process %d" % (data_send, destination_process))
#      print("data received is = %s" % data_received)
#
# if rank == 5: # 5 《=1 ； 5=》1
#      data_send = "b"
#      destination_process = 1
#      source_process = 1
#      # data_received = comm.recv(source=source_process)
#      # comm.send(data_send, dest=destination_process)
#      data_received=comm.sendrecv(data_send,dest=destination_process,source=source_process)
#      print("sending data %s to process %d" % (data_send, destination_process))
#      print("data received is = %s" % data_received)

# #广播 broadcast:comm.bcast
# from mpi4py import MPI
#
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
#
# if rank == 0:
#      variable_to_share = 100
# else:
#      variable_to_share = None
#
# variable_to_share = comm.bcast(variable_to_share, root=0)
# print("process = %d  variable shared = %d" % (rank, variable_to_share))

# #scatter:一个进程向不同的进程发送消息，每个进程收到的消息不一样
# from mpi4py import MPI
#
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
#
# # array_to_share = ["a","b","c","d","e","f","g","h","i","j"]
# if rank == 0:
#      array_to_share = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# else:
#      array_to_share = None
#
# recvbuf = comm.scatter(array_to_share, root=0)
# print("Process = %d  recvbuf = %s" % (rank, recvbuf))

# from mpi4py import MPI
#
# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# # print(size)
# rank = comm.Get_rank()
# data = "process %s" % rank
# print("start aaaa %s"%data)
# data = comm.gather(data, root=0) # root=0表示哪个进程要接收消息，这里是0号进程
# print("data bbbb%s"%data)
# # print(data)
# if rank == 0:
#      print("rank = %s receiving data from other process" % rank)
#      for i in range(1, size): #size是进程的个数
#           # data[i] = (i+1) ** 2
#           value = data[i]
#           print("process %s receiving %s from process %s" % (rank, value, i))
#      # print(data)


# #alltoall测试
# from mpi4py import MPI
# import numpy
#
# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()
#
# a_size = 1
#
# # print("numpy arange: %s" %numpy.arange(size, dtype=int))
# senddata = (rank + 1) * numpy.arange(size, dtype=int)
#
# recvdata = numpy.empty(size * a_size, dtype=int)
# print("senddata is %s , recvdata is %s" % (senddata, recvdata))
# # print("Recvdata is %s: , \n numpy.empty is %s" %(recvdata, numpy.empty(size * a_size, dtype=int)))
#
# comm.Alltoall(senddata, recvdata)
# print("process %s sending %s, receiving %s" % (rank, senddata, recvdata))

# julia.py

# """
# Demonstrates the usage of mpi4py.futures.MPIPoolExecutor.
#
# Run this with 1 processes like:
# $ mpiexec -n 1 -usize 17 python julia.py
# or 17 processes like:
# $ mpiexec -n 17 python -m mpi4py.futures julia.py
# """
# from mpi4py.futures import MPIPoolExecutor
#
# x0, x1, w = -2.0, +2.0, 640*2 #w:width
# y0, y1, h = -1.5, +1.5, 480*2 #h:height 960
# dx = (x1 - x0) / w
# dy = (y1 - y0) / h
#
# c = complex(0, 0.65)
#
# def julia(x, y):
#     z = complex(x, y)
#     n = 255
#     while abs(z) < 3 and n > 1:
#         z = z**2 + c
#         n -= 1
#     return n
#
# def julia_line(k):
#     print('k=',k) #0->959
#     line = bytearray(w)
#     y = y1 - k * dy
#     for j in range(w):
#         x = x0 + j * dx
#         line[j] = julia(x, y)
#     return line
#
#
# # if __name__ == '__main__':
# #     with MPIPoolExecutor() as executor:
# #         image = executor.map(julia_line, range(h)) #
# #         with open('julia.jpg', 'wb') as f:
# #             f.write(b'P5 %d %d %d\n' % (w, h, 255))
# #             for line in image:
# #                     f.write(line)
#
# from joblib import Parallel, delayed
# from math import sqrt
# aa=Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))
# print('aa:\n',aa)

# #读出模型的json文件并格式化显示出来
# import json
# with open("./payload2015_2017/models/10model.json",encoding="utf-8") as f:
#     result=json.load(f)
#     #print("result:\n",result,"\n",type(result))
#     result_str=json.dumps(result,indent=2)
#     print("result_str:\n",result_str)

# mydict={"a":{'tgt':"41.74444#41.80556#-87.82222#-87.74444#VAR"},"b":2}
# print(iter(mydict.values()).__next__()["tgt"])

# mylist1=[1,2,3]
# mylist2=[4,5,6]
# print(mylist1+mylist2)
import pandas as pd
# def df_setdiff(df1, df2):
#      df1__m = df1.apply(lambda x: tuple(x), axis=1)
#      print("df1__m:\n",df1__m,type(df1__m))
#      df2__m = df2.apply(lambda x: tuple(x), axis=1)
#      print("df2__m:\n", df2__m)
#      # bl=df1__['match'].isin(df2__['match']).values
#      print("~df1__m.isin(df2__m):\n",~df1__m.isin(df2__m))
#      df_ = df1[~df1__m.isin(df2__m)]
#
#      return df_
#
# df1=pd.DataFrame([[1.0,1],[2,2],[3,3]])
# print(df1)
# df2=pd.DataFrame([[1,1],[4,4],[5,5]])
# print(df2)
# print(df_setdiff(df1,df2))
#
# print((1.0,2.0)==(1.1,2))

# import numpy as np
# a,b=np.meshgrid([1,2,3],[4,5,6])
# print("a=",a)
# print("b=",b)
# print(np.meshgrid([1,2,3],[4,5,6]))

# print(a[0,:])
# print(b[:,0])
# print(np.array([True,True,False])*np.array([False,True,False]))
# print(a[0,:][18])
# def testFun():
#      return 1,2
# print(testFun()[1])

# ss=pd.date_range(start='2015-01-01',end='2018-12-31',freq='D')
# print('ss:\n',ss[:-1],len(ss[:-1]))
# import numpy as np
# print(np.linspace(41.5,42.05,10))
import numpy as np
import datetime
aa=pd.DatetimeIndex(['2015-01-01', '2015-01-02'])
np.savetxt("haha.haha",aa,fmt='%s')
