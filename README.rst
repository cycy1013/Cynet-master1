..use Cynet to predict stock close price
..all in stock2015_2017 directory
..the method is treating trade data as spacial factor
but now the result is confusing

for example:in 3modeluse120models#ALL#close.csv

tail content:
close,714,1,0.690375,0.309625,1,ALL,-1.0
close,715,1,0.676619,0.323381,1,ALL,-1.0
close,716,1,0.666565,0.333435,1,ALL,-1.0
close,717,0,0.6784,0.3216,1,ALL,-1.0
close,718,0,0.672881,0.327119,1,ALL,-1.0

in 3model_120_1_0.res content:

./stock2015_2017/models/3model,ALL,close,120,0.666663,0.333333,-1.0,1

1.why tpr,fpr is 0.333333,-1.0 ,come from where 

2.why fpr is negative

3.haven't seen valuable result in predicting stock close price

