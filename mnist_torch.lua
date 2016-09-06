require 'optim'
local nninit = require 'nninit'
mnist = require 'mnist'
require 'nn'


cmd = torch.CmdLine()
cmd:text('Basic Mnist example on Torch7')
cmd:text('Options')
cmd:option('-seed',1,'initial random seed')
cmd:option('-gpu',false,'boolean option')
cmd:option('-lr',0.001,'learning rate')
cmd:option('-batchsize',100,'batch size')
cmd:option('-epoch',10,'epoch')
cmd:text()
-- parse input params
opts = cmd:parse(arg)



-- seed for reproducebility
torch.manualSeed(opts.seed)


-- prepare the data
dataw = mnist.traindataset()
trainingset = {
   size = 60000,
   data = dataw.data[{{1,60000}}]:double(),
   label = dataw.label[{{1,60000}}]
}

-- build the model
model = nn.Sequential()
   :add(nn.Reshape(28*28))
   :add(nn.Linear(28*28, 64):init('weight', nninit.normal, 0, 0.001))
   :add(nn.ReLU())
   :add(nn.Linear(64, 10):init('weight', nninit.normal, 0, 0.001))
   :add(nn.LogSoftMax())


-- loss function
criterion = nn.ClassNLLCriterion()

print(model)
-- put them into gpu
if opts.gpu then
   require 'cunn'
   require 'cutorch'
   model = model:cuda()
   criterion = criterion:cuda()
   trainingset.data = trainingset.data:cuda()
   trainingset.label = trainingset.label:cuda()
end




-- get parameters and gradients
x, dl_dx = model:getParameters()

function train(batch_size)
   local trntime = 0
   local backtrntime= 0
   local current_loss = 0
   local count = 0
   for t = 1,trainingset.size,batch_size do
      local size = math.min(t + batch_size - 1, trainingset.size) - t
      local inputs,targets
      if opts.gpu then
	 inputs=torch.Tensor(size, 28, 28):cuda()
	 targets = torch.Tensor(size):cuda()
      else
	 inputs = torch.Tensor(size, 28, 28)
	 targets = torch.Tensor(size)
      end
      for i = 1,size do
	 local input = trainingset.data[i+t]
	 local target = trainingset.label[i+t]
	 inputs[i] = input
	 targets[i] = target
      end
      targets:add(1) -- to escape from class 0
      --  evaluate f(X) and df/dX
      local function feval(x_new)
	 -- reset data
	 if x ~= x_new then x:copy(x_new) end
	 dl_dx:zero()
	 -- forw and backw
	 tic = torch.tic()
	 local pred = model:forward(inputs)
	 local loss = criterion:forward(pred, targets)
	 local dloss = criterion:backward(model.output, targets)
	 model:backward(inputs,dloss)
	 trntime = trntime + torch.toc(tic)
	 return loss, dl_dx
      end
      -- fs is a table containing value of the loss function
      sgd_params = {
	 learningRate = opts.lr
      }
      _, fs = optim.sgd(feval,x,sgd_params)
      count = count + 1
      current_loss = current_loss + fs[1]
   end
   -- avg loss per batch
   return current_loss / count , trntime
end


-- training model
do
   local epoch = opts.epoch
   local batchsize= opts.batchsize
   for i = 1,epoch do
      local loss,epoch_time = train(batchsize)
      print(string.format('Epoch: %d Current loss: %4f Time:%.4f', i, loss,epoch_time))
   end
end
