# knn-python
A simple k-nearest neighbors (KNN) classifier algorithm implementation in python.

## Usage
#### initialize knn instance
```_knn = knn(k=8, distanceFunc=manhatten)```
#### fit model with trainingsdata
```_knn.fit(trainData,trainLabels)```
#### predicts labels for a given list of samples
```predictions = _knn.predict(samples)```
#### test model with testdata
```score = _knn.score(testData,testLabels)```

## Requirements
install requirements with: ```pip install -r requirements.txt```
- sklearn
- numpy
- matplotlib
