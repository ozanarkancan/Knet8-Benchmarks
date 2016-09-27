local nninit = require 'nninit'
local mnist = require 'mnist'
local cudnn = require 'cudnn'
require 'nn'
require 'cunn'

cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('--seed',1,'initial random seed')
cmd:option('--gpu',false,'boolean option')
cmd:option('--lr',0.1,'learning rate')
cmd:option('--batchsize',100,'batch size')
cmd:option('--epoch',1,'epoch')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- seed for reproducebility
torch.manualSeed(opt.seed)

-- prepare the data
dataw = mnist.traindataset()
dataq = mnist.testdataset()
trainingset = {
   size = 60000,
   data = dataw.data[{{1,60000}}]:double(),
   label = dataw.label[{{1,60000}}]
}

testset = {
   size = 10000,
   data = dataw.data[{{1,10000}}]:double(),
   label = dataw.label[{{1,10000}}]
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

scale(trainingset.data,0,1)
scale(testset.data,0,1)

trainingset.data=trainingset.data:view(trainingset.size,1,28,28) -- from 28x28 to 1x28x28
testset.data = testset.data:view(testset.size,1,28,28)

-- build the model
model = nn.Sequential()
-- cbfp stack - 1
model:add(cudnn.SpatialConvolution(1, 20, 5, 5, 1, 1, 0))--:init('weight', nninit.xavier))
model:add(nn.ReLU())
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
-- cbfp stack - 2
model:add(cudnn.SpatialConvolution(20, 50, 5, 5, 1, 1, 0))--:init('weight', nninit.xavier))
model:add(nn.ReLU())
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
-- hidden layer
model:add(nn.View(-1):setNumInputDims(3))
model:add(nn.Linear(800, 500))--:init('weight', nninit.xavier))
model:add(nn.ReLU())
model:add(nn.Linear(500, 10))--:init('weight', nninit.xavier))
model:add(nn.LogSoftMax())

-- loss function
criterion = nn.ClassNLLCriterion()
--print(model)


if opt.gpu then
   print("On GPU")
   model:cuda()
   criterion:cuda()
   trainingset.data = trainingset.data:cuda()
   trainingset.label = trainingset.label:cuda()
end

function eval(dataset, batch_size)
   local count = 0
   for i = 1,dataset.size,batch_size do
      local size = math.min(batch_size, trainingset.size - i + 1)
      local inputs = dataset.data[{{i,i+size-1},{},{},{}}]:cuda()
      local targets = dataset.label[{{i,i+size-1}}]:cuda()
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
      if opt.gpu then
	 inputs=torch.Tensor(size, 1, 28, 28):cuda()
	 targets = torch.Tensor(size):cuda()
      else
	 inputs = torch.Tensor(size, 1, 28, 28)
	 targets = torch.Tensor(size)
      end
      for i = 1,size do
	 local input = trainingset.data[i+t-1]
	 local target = trainingset.label[i+t-1]
	 inputs[i] = input
	 targets[i] = target
      end
      targets:add(1) -- classes through 1 to 10
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
      local accr = eval(testset,opt.batchsize)
      print(string.format('Epoch: %d Current loss: %4f Testset Accuracy:%.4f Time:%.4f', i, loss,accr,epoch_time))
   end
   print(string.format('Time:%.4f', total_time))
end