## SK Machine Learning
### Why

This is my adventure into machine learning. I don't currently have a future plan to use this program. Right now i want to make this as versatile as possible.

### How to use 
___
#### ***CSV Data Files*** 

A csv file is used for data. 

The training data should be formatted like this:

|feature1|feature2|feature3|feature4|target|
|--------|--------|--------|--------|------|
|data,   |data,   |data,   |data,   |data  |
|data,   |data,   |data,   |data,   |data  |
|data,   |data,   |data,   |data,   |data  |
...

The prediction Data should be formatted like this:\
The prediction data will have 1 less column.

|feature1|feature2|feature3|feature4|
|--------|--------|--------|--------|
|data,   |data,   |data,   |data    |
|data,   |data,   |data,   |data    |
|data,   |data,   |data,   |data    |
...

#### ***INI Config Files***
training-config.ini is the configurations for skm-learn. prediction-config.ini is used to configure skm-predict. You can point to your data in these files. This is also were you will set the amount of features.

#### ***Train a model***
 After getting your data into csv files and setting the ini configs you can run skm-learn to train a model. Once a model.json file is created (the name might differ depending on configurations), skm-predict can be run to predict a true or false categorizing.  