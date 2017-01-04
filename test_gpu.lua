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

net = torch.load("model.t7")
print ('Model loaded')


img_counter = 1
row_counter = 15
col_counter = 16
epochs_counter=0

rows = 584
cols = 565

---
--- Testing
---

img_counter = 2
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

until total == 5000

print ("\n Correct predictions: ")
print (correct)

cutorch.synchronize()






