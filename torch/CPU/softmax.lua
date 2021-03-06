-- sample run on gpu: th softmax.lua 
local nninit = require 'nninit'
mnist = require 'mnist'
require 'nn'

cmd = torch.CmdLine()
cmd:text('Basic Mnist example on Torch7')
cmd:text('Options')
cmd:option('--seed',1,'initial random seed')
cmd:option('--lr',0.5,'learning rate')
cmd:option('--batchsize',100,'batch size')
cmd:option('--epoch',10,'epoch')
cmd:text()

-- use single thread for comparability
torch.setnumthreads(1)

-- parse input params
opt = cmd:parse(arg)

-- seed for reproducibility
torch.manualSeed(opt.seed)

-- prepare the data
dataw = mnist.traindataset()
dataq = mnist.testdataset()

trainingset = {
   size = 60000,
   data = dataw.data[{{1,60000}}]:float(),
   label = dataw.label[{{1,60000}}]:float()
}

testset = {
   size = 10000,
   data = dataq.data[{{1,10000}}]:float(),
   label = dataq.label[{{1,10000}}]:float()
}

function scale(data, min, max)
   range = max - min
   dmin = data:min()
   dmax = data:max()
   drange = dmax - dmin
   data:add(-dmin)
   data:mul(range)
   data:mul(1/drange)
   data:add(min)
end

-- normalize mnist data
scale(trainingset.data,0,1)
scale(testset.data,0,1)

-- build the model
local linearLayer = nn.Linear(28*28,10)
linearLayer.weights = torch.rand(28*28,10)*0.1
linearLayer.bias = torch.zeros(10)
model = nn.Sequential()
   :add(nn.Reshape(28*28))
   :add(linearLayer)
   :add(nn.LogSoftMax())

-- loss function
criterion = nn.ClassNLLCriterion()

model:float()
criterion:float()
function eval(dataset, batch_size)
   local count = 0
   for i = 1,dataset.size,batch_size do
      local size = math.min(batch_size, trainingset.size - i + 1)
      local inputs,targets
      inputs = torch.FloatTensor(size, 28, 28)
      targets = torch.FloatTensor(size)
      for t = 1,size do
         local input = trainingset.data[i+t-1]
         local target = trainingset.label[i+t-1]
         inputs[t] = input
         targets[t] = target
      end
      local outputs = model:forward(inputs)
      outputs = outputs:float()
      targets = targets:long()
      local _, indices = torch.max(outputs, 2)
      indices:add(-1)
      local guessed_right = indices:eq(targets):sum()
      count = count + guessed_right
   end
   return count / dataset.size
end

function train(batch_size)
   local trntime = 0
   local current_loss = 0
   local count = 0
   for t = 1,trainingset.size,batch_size do
      local size = math.min(batch_size, trainingset.size- t + 1)
      local inputs,targets
      inputs = torch.FloatTensor(size, 28, 28)
      targets = torch.FloatTensor(size)
      for i = 1,size do
	 local input = trainingset.data[i+t-1]
	 local target = trainingset.label[i+t-1]
	 inputs[i] = input
	 targets[i] = target
      end
      targets:add(1) -- to escape from class 0
      local tic = torch.tic()
      local loss = criterion:forward(model:forward(inputs),targets)
      model:zeroGradParameters()
      model:backward(inputs,criterion:backward(model.output,targets))
      model:updateParameters(opt.lr)
      trntime = trntime + torch.toc(tic)
      current_loss = current_loss + loss
      count = count + 1
   end
   -- avg loss per batch
   return current_loss / count , trntime
end

-- training model
do
   total_time = 0
   local initaccr = eval(testset,opt.batchsize)
   print(string.format('Epoch: %d Initial Testset Accuracy: %.4f',0,initaccr ))   
   for i = 1,opt.epoch do
      local loss,epoch_time = train(opt.batchsize)
      total_time = total_time + epoch_time
      local eval_res = eval(testset,opt.batchsize)
      print(string.format('Epoch: %d Loss: %.4f Testset Accuracy: %.4f Time:%.4f', i,loss,eval_res,epoch_time))
   end
   print(string.format('Time:%.4f', total_time))
end
