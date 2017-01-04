require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'gnuplot'
require 'math'
require 'image'
require 'cutorch'
require 'cunn'

torch.manualSeed(1)
---
--- Load dataset
---

training_img = torch.Tensor(20, 3, 584, 565)
for i = 1, 20 do
	training_img[i]=image.load("training/images/" .. tostring(i) .. ".JPEG" ,3)
end

truth = torch.Tensor(20, 1, 584, 565)
for i = 1, 20 do
	truth[i]=image.load("training/truth/" .. tostring(i) .. ".JPEG" ,1)
end

mask = torch.Tensor(20, 1, 584, 565)
for i = 1, 20 do
	mask[i]=image.load("training/mask/" .. tostring(i) .. ".JPEG" ,1)
end


training_img = training_img:cuda()
truth = truth:cuda()
mask = mask:cuda()

-- image.display(training_img[4])
-- image.display(truth[4])
-- image.display(mask[4])

print (training_img:size()[1] .. ' images loaded')

---
--- Model
---

net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 64, 4, 4)) -- 3 input image channel, 64 output channels, 4*4 convolution kernel
net:add(nn.ReLU())                       -- non-linearity 

net:add(nn.SpatialConvolution(64, 64, 4, 4, 1, 1, 1, 1)) 
net:add(nn.ReLU())  
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

net:add(nn.SpatialConvolution(64, 64, 4, 4, 1, 1, 1, 1)) 
net:add(nn.ReLU())  
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

net:add(nn.View(64*6*6))                    -- reshapes from a 3D tensor of 64*6*6 into 1D tensor of 64*6*6
net:add(nn.Linear(64*6*6, 256))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())                       -- non-linearity 

net:add(nn.Dropout(0.7))
net:add(nn.Linear(256, 2))
net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems

criterion = nn.ClassNLLCriterion()

net = net:cuda()
criterion = criterion:cuda()

print('Santara-Net\n' .. net:__tostring());

---
---Initialzie wts
---

function w_init_xavier(fan_in, fan_out)
   return math.sqrt(2/(fan_in + fan_out))
end

local function w_init(net)
   -- choose initialization method
   local method = w_init_xavier

   -- loop over all convolutional modules
   for i = 1, #net.modules do

      local m = net.modules[i]
      if m.__typename == 'nn.SpatialConvolution' then
         m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
      elseif m.__typename == 'cudnn.SpatialConvolution' then
         m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
      elseif m.__typename == 'nn.Linear' then
         m:reset(method(m.weight:size(2), m.weight:size(1)))          
      end

      if m.bias then
         m.bias:zero()
      end
   end
   return net
end

net = w_init(net)


---
--- Training
---

parameters, gradParameters = net:getParameters()

img_counter = 1
row_counter = 15
col_counter = 16
epochs_counter=0

rows = 584
cols = 565

feval = function(x)
  --print ("feval called")
  if x ~= parameters then
    parameters:copy(x)
  end

  valid_input = true

  repeat
  	row_counter = row_counter + 1
  	if row_counter >= rows - 16 then
  		row_counter = 16
  		col_counter = col_counter + 1
  	end

  	if col_counter >= cols - 16 then
  		col_counter = 16
  		img_counter = img_counter + 1
  	end

  	if img_counter > 20 then
  		img_counter = 1
      epochs_counter = epochs_counter + 1
  	end

  	---print ("checking for img_counter= " .. img_counter .. " col_counter= " .. col_counter .. " row_counter= " .. row_counter)

  	valid_input = true
  	for i = row_counter - 15, row_counter + 16 do
  		for j = col_counter -15 , col_counter + 16 do
  			if mask[img_counter][1][i][j] == 0 then
  				valid_input = false
  			end
        if valid_input == false then break end
  		end
      if valid_input == false then break end
  	end
  until valid_input == true

  --print ("found valid input patch")

  batch_inputs = training_img[img_counter][{ {1,3}, {row_counter-15,row_counter+16}, {col_counter-15,col_counter+16}}]
  batch_targets = truth[img_counter][{1, row_counter, col_counter}] + 1

  gradParameters:zero()

  batch_outputs = net:forward(batch_inputs)
  batch_loss = criterion:forward(batch_outputs, batch_targets)
  dloss_doutput = criterion:backward(batch_outputs, batch_targets)
  net:backward(batch_inputs, dloss_doutput)

  return batch_loss, gradParameters
end

opt = {}         -- these options are used throughout
opt.optimization = 'rmsprop'
opt.batch_size = 20
opt.train_size = 20

optimState = {
    learningRate = 1e-3,			-- stores a lua table with the optimization algorithm's settings, and state during iterations
  }
optimMethod = optim.rmsprop		-- stores a function corresponding to the optimization routine


losses = {}
epochs = opt.epochs
iterations = 10000000



for i = 1, iterations do
	_, minibatch_loss = optimMethod(feval, parameters, optimState)

	if i % 100 == 0 then -- don't print *every* iteration, this is enough to get the gist
      print(string.format("minibatches processed: %6s, loss = %6.6f", i, minibatch_loss[1]))
  end

  if i%1000 ==0 then
    print ("Training on image: " .. tostring(img_counter))
  end

  if i%1000 == 0 then
    torch.save("model.t7", net)
  end

  	losses[#losses + 1] = minibatch_loss[1] -- append the new loss

  if epochs_counter >=1 then break end
end

cutorch.synchronize()

-- Turn table of losses into a torch Tensor, and plot it
gnuplot.plot({
  torch.range(1, #losses),        -- x-coordinates for data to plot, creates a tensor holding {1,2,3,...,#losses}
  torch.Tensor(losses),           -- y-coordinates (the training losses)
  '-'})


---
--- Testing
---

img_counter = 1
row_counter = 15
col_counter = 16

correct=0
total=0

print ("Testing ...")

repeat

	if total % 50 ==0 then
		print (total .. " ....")
	end

	repeat
  		row_counter = row_counter + 1
  		if row_counter >= rows - 16 then
  			row_counter = 16
  			col_counter = col_counter + 1
  		end

  		if col_counter >= cols - 16 then
  			col_counter = 16
  			img_counter = img_counter + 1
  		end

  		if img_counter > 20 then
  			img_counter = 1
  		end

  		---print ("checking for img_counter= " .. img_counter .. " col_counter= " .. col_counter .. " row_counter= " .. row_counter)

  		valid_input = true
  		for i = row_counter - 15, row_counter + 16 do
  			for j = col_counter -15 , col_counter + 16 do
  				if mask[img_counter][1][i][j] == 0 then
  					valid_input = false
  				end
          if valid_input == false then break end
  			end
        if valid_input == false then break end
  		end
  	until valid_input == true

  	batch_inputs = training_img[img_counter][{ {1,3}, {row_counter-15,row_counter+16}, {col_counter-15,col_counter+16}}]
  	batch_targets = truth[img_counter][{1, row_counter, col_counter}] + 1

  	batch_outputs = net:forward(batch_inputs)

  	total=total+1
  	_, argm = torch.max(batch_outputs, 1)

    --print (tostring(batch_targets) .. " != " .. tostring(argm))

  	if batch_targets > 1.2 then
  		correct = correct + (argm-1)
  	else
  		correct = correct + (2-argm)
  	end

until total == 500

print ("\n Correct predictions: ")
print (correct)

cutorch.synchronize()






