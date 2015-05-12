-- Character Recognition for Kaggle competition
	-- Eli Gabay and Kyle Crowley

-- Libraries
require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'csvigo'
require 'string'

-- Global variables
globalTrainSize = 6283

-- Function declarations

-- Iterate through a table searching for element as a value
function table.contains(table, element)
  for _, value in pairs(table) do
    if value == element then
      return true
    end
  end
  return false
end
-- Iterate through a table searching for the index containing element
function table.findIndex(table, element)
  for i,value in pairs(table) do
    if tonumber(value) == tonumber(element) then
      return i
    end
  end
  return nil
end

-- Directory names
trainDir = 'TrainDataResized/'
testDir = 'TestDataResized/'

-- Classes
classes = {}

--------------
-- DATA
--------------
print('==> Loading train and test set')

Filenames = {}
ClassIds = {}

-- Create filenames
for i=1,6283 do
	if i < 1001 then
		table.insert(Filenames,trainDir .. 'Set1/' .. i .. '.jpg')
	elseif i < 2001 then
		table.insert(Filenames,trainDir .. 'Set2/' .. i .. '.jpg')
	elseif i < 3001 then
		table.insert(Filenames,trainDir .. 'Set3/' .. i .. '.jpg')
	elseif i < 4001 then
		table.insert(Filenames,trainDir .. 'Set4/' .. i .. '.jpg')
	elseif i < 5001 then
		table.insert(Filenames,trainDir .. 'Set5/' .. i .. '.jpg')
	elseif i < 6001 then
		table.insert(Filenames,trainDir .. 'Set6/' .. i .. '.jpg')
	elseif i < 7001 then
		table.insert(Filenames,trainDir .. 'Set7/' .. i .. '.jpg')
	end
end

csvfName = 'trainLabels.csv'
query = csvigo.load({path=csvfName,verbose=false,mode='query'})
temp = query('vars')[1]

for i=1,#query('all')[temp] do
	
	-- Take line
	line = query('all')[temp][i]
	len = string.len(line)

	table.insert(ClassIds,line)

	if not table.contains(classes,line) then
		table.insert(classes,line)
	end
	
end

for i=1,#classes do
	classes[i] = '' .. string.byte(classes[i])
end

-- Random split the train set to 5000 Train, 1283 Test subsets 
-- (Test set in this path is for competition submissions)
print('==> Randomly splitting training data')
shuffle = torch.randperm(globalTrainSize)
shuffledTrainfNames = {}
shuffledTrainClassIds = {}

shuffledTestfNames = {}
shuffledTestClassIds = {}
-- 5000 Train subset
for i=1,5000 do
	table.insert(shuffledTrainfNames,Filenames[shuffle[i]])
	table.insert(shuffledTrainClassIds,ClassIds[shuffle[i]])
end
-- 1283 Test subset
for i=5001,6283 do
	table.insert(shuffledTestfNames,Filenames[shuffle[i]])
	table.insert(shuffledTestClassIds,ClassIds[shuffle[i]])
end

-- Read in test images
print('==> Uploading images for training & testing')
shuffledTrainImages = {}
shuffledTestImages = {}
-- Images for training
for i=1,5000 do
	imageData = image.load(shuffledTrainfNames[i])
	table.insert(shuffledTrainImages,imageData)
end
-- Images for testing
for i=1,1283 do
	imageData = image.load(shuffledTestfNames[i])
	table.insert(shuffledTestImages,imageData)
end

-- Convert images to YUV 
print('	==> Colorspace RGB -> YUV')
trainImages = {}
testImages = {}
for i=1,5000 do
	local removed = table.remove(shuffledTrainImages, 1)
	table.insert(trainImages,image.rgb2yuv(removed))
end
for i=1,1283 do
	local removed = table.remove(shuffledTestImages, 1)
	table.insert(testImages,image.rgb2yuv(removed))
end

--print('	==> Normalize each channel')
--print('	==> Spacial contrastive normalization')

------------------
-- MODEL (ConvNet)
------------------
print('==> Creating model')
-- Parameters
noutputs = 62

nfeats = 3
width = 20
height = 20
ninputs = nfeats*width*height

nhiddens = ninputs/2
nstates  = {40,40,80}
filtsize = 3
poolsize = 2
normkernel = image.gaussian1D(3)

-- Container
model = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

-- stage 3 : standard 2-layer neural network
model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
model:add(nn.Tanh())
model:add(nn.Linear(nstates[3], noutputs))

-----------
-- LOSS
-----------
print('==> Creating loss')

model:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()

confusion = optim.ConfusionMatrix(classes)

trainLogger = optim.Logger('train.log')
testLogger = optim.Logger('test.log')

parameters, gradParameters = model:getParameters()

sgdconf = {
	learningRate = 1e-3,
	weightDecay = 0,
	momentum = 0,
	learningRateDecay = 1e-7
	}
optimMethod = optim.sgd

---------
-- TRAIN
---------
plot = true
print('==> defining training procedure')
batchSize = 64
trsize = 5000

function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   for t = 1,trsize,batchSize do
      -- disp progress
      xlua.progress(t, trsize)

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+batchSize-1,trsize) do
         -- load new sample
         local input = trainImages[shuffle[i]]
         local target = shuffledTrainClassIds[shuffle[i]]

         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
									local trg = string.byte(targets[i])
									ind = table.findIndex(classes,trg)
		
									local inp = torch.Tensor(3,20,20):copy(inputs[i])

                          local output = model:forward(inp)
                          local err = criterion:forward(output, ind)

                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, ind)
                          model:backward(inp, df_do)

                          -- update confusion
                          confusion:add(output, ind)
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
      if optimMethod == optim.asgd then
        _,_,average = optimMethod(feval, parameters, sgdconf)
      else
         optimMethod(feval, parameters, sgdconf)
      end
   end

   -- time taken
   time = sys.clock() - time
   time = time / trsize
   print("\n==> time to learn 1 sample = " .. (time*1000) .. ' ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   local filename = 'model.net'
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end

---------
-- TEST
---------
print('==> defining test procedure')
tesize = 1283
-- test function
function test()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1,tesize do
      -- disp progress
      xlua.progress(t, tesize)

      -- get new sample
      local input = testImages[t]

      local target = shuffledTestClassIds[t]
		local trg = string.byte(target)
		ind = table.findIndex(classes,trg)
--print(ind)
      -- test sample
      local pred = model:forward(input)
      confusion:add(pred,ind)
   end

   -- timing
   time = sys.clock() - time
   time = time / tesize
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
end

while true do
	train()
	test()
end

