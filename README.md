# charRNN
Recurrent neural networks that are fast, easy to use, and can reproduce [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

The rnn code, text parsing code, and shakespeare data is taken from [https://github.com/martin-gorner/tensorflow-rnn-shakespeare](https://github.com/martin-gorner/tensorflow-rnn-shakespeare), I just cleaned up the code to make it easier to use

The movies folder contains the dataset I got from [here](https://github.com/suriyadeepan/practical_seq2seq)

I also combined all linux source code files into a single .txt file, you can download that [here](https://drive.google.com/file/d/1nOsvza-zcQh60FLZHsoU4qbJpbUNS1me/view?usp=sharing) 


To use this, you do:

```python
import charRNN
model = charRNN.CharRNN("test")
data = model.loadData("shakespeare/")
model.fit(data) # press ctrl-c after 2-3 hours of training, this autosaves periodically
model.generate(1000, seed="K") # generate a string of 1000 characters, starting with the letter "K"
```

In more detail:

```python
name = "test" # this can be anything, is just used for storing checkpoints (so this model's checkpoints will be stored in ./checkpoints/test)
# feel free to have more than one of these in memory, simultaneous models are supported (but you should probably keep the names different so they don't overwrite each other's saves)
model = charRNN.CharRNN(name)

# optional params (default values are listed here):
seqLen = 30 # this is how long the sequences are unrolled in training. Your neural network can only be expected to learn patterns consisting of seqLen characters or less, so the longer this is, the more long term dependencies your network can learn. However sometimes it'll still figure out larger long term dependencies than seqLen anyway, and 30 (the default) typically gives pretty good results.
internalSize = 512 # how large each hidden layer is. It is best to keep this a power of two for the optimizations to work properly
nLayers = 3 # how deep your rnn is
# for example you can do
model = charRNN.CharRNN(name, seqLen=50, internalSize=512, nLayers=3)

# call loadData with a directory containing .txt files and the model will parse them into a dataset it can use for training
data = model.loadData("shakespeare/")

# train the model
# it will print every 50 batches, generate text every 150 batches, and save a checkpoint every 500 batches. You should expect to get fairly good results after 1-2 hours, but for very good results you should wait 4-5 hours (these times might be different depending on how good your computer is)
# feel free to press ctrl-c to stop this (just only press ctrl-c once, don't spam it), it'll detect that and save your model
model.fit(data)

# generate a string of length 1000
model.generate(1000)

# you can also manually save your model, it'll print out where it saved. For example, after doing some training, this will print out
# 'Saved file: checkpoints/test/rnn_train_test-102000'
# if you look in there you will see four files with that name and extensions .data-00000-of-00001, .index, .json, .meta. All four of these files are needed to load a model.
model.save()

# to load a model
model2 = charRNN.CharRNN("test2", prevPath="checkpoints/test/rnn_train_test-102000")
```
