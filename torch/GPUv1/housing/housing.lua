-- Demonstrate a linear regression model using housing data from the UCI Machine Learning Repository
-- Data reader part has been adapted from  https://github.com/ronanmoynihan/biased-regression
-- sample run on gpu th housing.lua
require 'nn'
require 'cunn'
require 'cutorch'

local nninit = require 'nninit'

local data_loader = require 'data'
local data_file = 'data/Boston.th'
local data_train_percentage = 100


--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Demonstrate a linear regression model using housing data from the UCI Machine Learning Repository')
cmd:text(' Sample usage: th housing.lua --gpu')
cmd:text('Options:')
cmd:option('--seed',1,'initial random seed')
cmd:option('--datapath','data/Boston.th','housing data path')-- The Boston.csv file has been converted to torch.
cmd:option('--lr',0.08,'learning rate')
cmd:option('--batchsize',506,'batch size')
cmd:option('--epoch',10000,'epoch')
cmd:option('--training_percentage',100,'percent of the data that is used to train the model')
cmd:text()

collectgarbage()
collectgarbage("setpause",100000000)

local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-- load data
data = data_loader.load_data(opt.datapath, opt.training_percentage)
data.train_data=data.train_data:float()
data.train_targets=data.train_targets:float()
-- model
local linearLayer = nn.Linear(data.train_data:size(2),1)
linearLayer.weights = torch.rand(data.train_data:size(2),1)* 0.1
linearLayer.bias = torch.zeros(1)
model = nn.Sequential():add(linearLayer)
-- criterion
local criterion = nn.MSECriterion()

model = model:cuda()
criterion = criterion:cuda()
data.train_data = data.train_data:cuda()
data.train_targets = data.train_targets:cuda()

-- train
do
total_time = 0
for i = 1,opt.epoch do
   local tic = torch.tic()
   local pred = model:forward(data.train_data)
   local loss = criterion:forward(pred,data.train_targets)
   model:zeroGradParameters()
   model:backward(data.train_data,criterion:backward(pred,data.train_targets))
   model:updateParameters(opt.lr)
   total_time = total_time +   torch.toc(tic)
end
print(string.format('Total time:%.4f',total_time))
end
