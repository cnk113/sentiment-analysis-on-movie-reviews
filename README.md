# Semantic Analysis of Movie Reviews

Carson Burr (cwburr@ucsc.edu)  
Chang Kim (cnkim@ucsc.edu)  
Bryan Tor (btor@ucsc.edu)


## extractModel.py
---
train the model. outputs model.joblib to be used in testModel.py  
```
Usage: python testModel.py yourtrainset.csv  
Usage: python testModel.py # uses 'train.csv' if no file specified
```

## birnn.py
---
Uses bidirectional RNN with both recurrent and normal dropout.
Uses Nesterov's adaptive momentum 
Train time with current parameters on a Google TPU ~90 minutes
Achieves ~65% accuracy, but model diverged.
Interesting problem due to either too high of a learning rate or the categorical cross entropy cost function

## model.joblib
---
a serialized python object of a sklearn classifier


## testModel.py
---
test on a csv. outputs prediction.csv  
```
Usage: python testModel.py yourtestset.csv  
Usage: python testModel.py # uses 'testset_1.csv' if no file specified
```
