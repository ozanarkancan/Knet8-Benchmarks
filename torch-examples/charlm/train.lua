-- This code is the simplified version of torch-rnn repository
-- All the code has been taken from https://github.com/jcjohnson/torch-rnn
-- And modified accordingly.
-- sample usage : 
-- theano_py scripts/preprocess.py --input_txt data/19.txt --output_h5 data/my_data.h5 --output_json data/my_data.json
-- th train.lua -input_h5 data/my_data.h5 -input_json data/my_data.json  -num_layers 1 -rnn_size 256 -speed_benchmark 1
require 'torch'
require 'nn'
require 'optim'

require 'LanguageModel'
require 'util.DataLoader'

local utils = require 'util.utils'
local unpack = unpack or table.unpack

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-input_h5', 'data/my_data.h5')
cmd:option('-input_json', 'data/my_data.json')
cmd:option('-batch_size', 128)
cmd:option('-seq_length', 100)
cmd:option('--seed',1,'initial random seed')
-- Model options
cmd:option('-init_from', '')
cmd:option('-reset_iterations', 1)
cmd:option('-model_type', 'lstm')
cmd:option('-wordvec_size', 256)
cmd:option('-rnn_size', 256)
cmd:option('-num_layers', 1)
cmd:option('-dropout', 0)
cmd:option('-batchnorm', 0)
-- Optimization options
cmd:option('-max_epochs', 1)
cmd:option('-learning_rate', 4.0)
cmd:option('-grad_clip', 3.0)

-- Output options
cmd:option('-print_every', 1)
cmd:option('-speed_benchmark', 0)
-- Backend options
cmd:option('-gpu', 0)
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-- Set up GPU stuff

local dtype = 'torch.FloatTensor'
if opt.gpu >= 0  then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  dtype = 'torch.CudaTensor'
  print(string.format('Running with CUDA on GPU %d', opt.gpu))
end

-- Initialize the DataLoader and vocabulary
local loader = DataLoader(opt)
local vocab = utils.read_json(opt.input_json)
local idx_to_token = {}
for k, v in pairs(vocab.idx_to_token) do
  idx_to_token[tonumber(k)] = v
end

-- Initialize the model and criterion
local opt_clone = torch.deserialize(torch.serialize(opt))
opt_clone.idx_to_token = idx_to_token

local start_i = 0
local model = nn.LanguageModel(opt_clone):type(dtype)

local params, grad_params = model:getParameters()
local crit = nn.CrossEntropyCriterion():type(dtype)

-- Set up some variables we will use below
local N, T = opt.batch_size, opt.seq_length
mainTotal=0
-- Loss function that we pass to an optim method
local function f(w)
  assert(w == params)
  grad_params:zero()

  -- Get a minibatch and run the model forward, maybe timing it
  local timer
  local x, y = loader:nextBatch('train')
  x, y = x:type(dtype), y:type(dtype)
  if opt.speed_benchmark == 1 then
    if cutorch then cutorch.synchronize() end
    timer = torch.Timer()
  end
  local scores = model:forward(x)

  -- Use the Criterion to compute loss; we need to reshape the scores to be
  -- two-dimensional before doing so. Annoying.
  local scores_view = scores:view(N * T, -1)
  local y_view = y:view(N * T)
  local loss = crit:forward(scores_view, y_view)

  -- Run the Criterion and model backward to compute gradients, maybe timing it
  local grad_scores = crit:backward(scores_view, y_view):view(N, T, -1)
  model:backward(x, grad_scores)

  if opt.grad_clip > 0 then
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end
  --if timer then
    if cutorch then cutorch.synchronize() end
    local time = timer:time().real
    --print('Forward / Backward pass took ', time)
  --end
  mainTotal= mainTotal + time
  return loss, grad_params
end

-- Train the model!
local optim_config = {learningRate = opt.learning_rate}
local num_train = loader.split_sizes['train']
local num_iterations = opt.max_epochs * num_train
--print(num_train)
--print(num_iterations)
model:training()
for i = start_i + 1, num_iterations do
  local epoch = math.floor(i / num_train) + 1
  -- Check if we are at the end of an epoch
  if i % num_train == 0 then
    model:resetStates() -- Reset hidden states
  end
  -- Take a gradient step and maybe print
  -- Note that adam returns a singleton array of losses
  local _, loss = optim.sgd(f, params, optim_config)
 
  if opt.print_every > 0 and i % opt.print_every == 0 then
    local float_epoch = i / num_train 
    local msg = 'Epoch %.2f / %d, i = %d / %d, loss = %f time= %.4f'
    local args = {msg, float_epoch, opt.max_epochs, i, num_iterations, loss[1], mainTotal}
    print(string.format(unpack(args)))
  end
end
