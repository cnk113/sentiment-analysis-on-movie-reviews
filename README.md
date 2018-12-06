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