-- Demonstrate a linear regression model using housing data from the UCI Machine Learning Repository
-- Data reader part has been adapted from  https://github.com/ronanmoynihan/biased-regression
-- sample run on gpu th housing.lua --gpu
require 'nn'
local nninit = require 'nninit'

local data_loader = require 'data'
local data_file = 'data/Boston.th'
local data_train_percentage = 100

-- use single thread for comparability
torch.setnumthreads(1)

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

local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-- load data
data = data_loader.load_data(opt.datapath, opt.training_percentage)
data.train_data=(torch.FloatTensor(506,13):copy(data.train_data))
data.train_targets=(torch.FloatTensor(506,1):copy(data.train_targets))

-- model
local linearLayer = nn.Linear(data.train_data:size(2),1)
linearLayer.weights = torch.rand(data.train_data:size(2),1)* 0.1
linearLayer.bias = torch.zeros(1)
model = nn.Sequential():add(linearLayer)

-- criterion
local criterion = nn.MSECriterion()
model:float()
criterion:float()

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
   --print(string.format('Epoch: %d Current loss: %4f', i, loss))
end
print(string.format('Total time:%.4f',total_time))
end
