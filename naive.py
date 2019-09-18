# -*- coding: utf-8 -*-
"""
@author: Ramani
"""

import math
import pandas as pd
import numpy as np
 
data = pd.read_csv('./data/train numbers.csv')

"""
The following function takes our unsorted data and sorts them into all discrete variables. 
The variables that were already discrete in the initial data were enumerated by hand. To discretize
the continuos variables each variable is split up into 4 buckets based on the quartiles of the range
of values that the variable could have. These values are then hardcoded into a numpy array and the
for that attribute is placed into buckets using the search sorted method in numpy. The new buckets are
added into a separate column and the old one is deleted. As mentioned in our write up the Price
attribute is broken up into 16 buckets not 4.
"""
def sortdata():
    arr = np.array([78, 156, 234, 314])
    data['SortedLotFrontage'] = arr.searchsorted(data.LotFrontage)
    data.drop('LotFrontage', 1, inplace=True)
    arr = np.array([53811, 53811*2,53811*3, 215245])
    data['SortedLotArea'] = arr.searchsorted(data.LotArea)
    data.drop('LotArea', 1, inplace=True)
    arr = np.array([1907, 1942 ,1977, 2011])
    data['SortedYearBuilt'] = arr.searchsorted(data.YearBuilt)
    data.drop('YearBuilt', 1, inplace=True)
    arr = np.array([1965, 1980, 1995, 2011])
    data['SortedYearRemodAdd'] = arr.searchsorted(data.YearRemodAdd)
    data.drop('YearRemodAdd', 1, inplace=True)
    arr = np.array([400, 400*2, 400*3, 1601])
    data['SortedMasVnrArea'] = arr.searchsorted(data.MasVnrArea)
    data.drop('MasVnrArea', 1, inplace=True)
    arr = np.array([1411, 1411*2,1411*3, 5644])
    data['SortedBsmtFinSF1'] = arr.searchsorted(data.BsmtFinSF1)
    data.drop('BsmtFinSF1', 1, inplace=True)
    arr = np.array([369, 369*2,369*3, 1474])
    data['SortedBsmtFinSF2'] = arr.searchsorted(data.BsmtFinSF2)
    data.drop('BsmtFinSF2', 1, inplace=True)
    arr = np.array([584, 584*2,584*3, 2336])
    data['SortedBsmtUnfSF'] = arr.searchsorted(data.BsmtUnfSF)
    data.drop('BsmtUnfSF', 1, inplace=True)
    arr = np.array([1523, 1523*2,1523*3, 1523*4])
    data['SortedTotalBsmtSF'] = arr.searchsorted(data.TotalBsmtSF)
    data.drop('TotalBsmtSF', 1, inplace=True)
    arr = np.array([1173, 1173*2,1173*3, 1173*4])
    data['Sorted1stFlrSF'] = arr.searchsorted(data['1stFlrSF'])
    data.drop('1stFlrSF', 1, inplace=True)
    arr = np.array([517, 517*2,517*3, 517*4])
    data['Sorted2ndFlrSF'] = arr.searchsorted(data['2ndFlrSF'])
    data.drop('2ndFlrSF', 1, inplace=True)
    arr = np.array([143, 143*2,143*3, 143*4])
    data['SortedLowQualFinSF'] = arr.searchsorted(data.LowQualFinSF)
    data.drop('LowQualFinSF', 1, inplace=True)
    arr = np.array([1411, 1411*2,1411*3, 1411*4])
    data['SortedGrLivArea'] = arr.searchsorted(data.GrLivArea)
    data.drop('GrLivArea', 1, inplace=True)
    arr = np.array([2,4,6,9])
    data['SortedBedroomAbvGr'] = arr.searchsorted(data.BedroomAbvGr)
    data.drop('BedroomAbvGr', 1, inplace=True)
    arr = np.array([1,2,3,4])
    data['SortedBsmtFullBath'] = arr.searchsorted(data.BsmtFullBath)
    data.drop('BsmtFullBath', 1, inplace=True)
    arr = np.array([1, 2, 3])
    data['SortedBsmtHalfBath'] = arr.searchsorted(data.BsmtHalfBath)
    data.drop('BsmtHalfBath', 1, inplace=True)
    arr = np.array([1,2,3,4])
    data['SortedFullBath'] = arr.searchsorted(data.FullBath)
    data.drop('FullBath', 1, inplace=True)
    arr = np.array([1,2,3,4])
    data['SortedHalfBath'] = arr.searchsorted(data.HalfBath)
    data.drop('HalfBath', 1, inplace=True)
    arr = np.array([2,4,6,9])
    data['SortedKitchenAbvGr'] = arr.searchsorted(data.KitchenAbvGr)
    data.drop('KitchenAbvGr', 1, inplace=True)
    arr = np.array([4,8,12,16])
    data['SortedTotRmsAbvGrd'] = arr.searchsorted(data.TotRmsAbvGrd)
    data.drop('TotRmsAbvGrd', 1, inplace=True)
    arr = np.array([4,8,12,16])
    data['SortedFireplaces'] = arr.searchsorted(data.Fireplaces)
    data.drop('Fireplaces', 1, inplace=True)
    arr = np.array([1907, 1942 ,1977, 2011])
    data['SortedGarageYrBlt'] = arr.searchsorted(data.GarageYrBlt)
    data.drop('GarageYrBlt', 1, inplace=True)
    arr = np.array([1, 2,3, 5])
    data['SortedGarageCars'] = arr.searchsorted(data.GarageCars)
    data.drop('GarageCars', 1, inplace=True)
    arr = np.array([355, 355*2,355*3, 355*4])
    data['SortedGarageArea'] = arr.searchsorted(data.GarageArea)
    data.drop('GarageArea', 1, inplace=True)
    arr = np.array([215, 215*2,215*3, 215*4])
    data['SortedWoodDeckSF'] = arr.searchsorted(data.WoodDeckSF)
    data.drop('WoodDeckSF', 1, inplace=True)
    arr = np.array([137, 137*2,137*3, 137*4])
    data['SortedOpenPorchSF'] = arr.searchsorted(data.OpenPorchSF)
    data.drop('OpenPorchSF', 1, inplace=True)
    arr = np.array([138, 138*2,138*3, 138*4])
    data['SortedEnclosedPorch'] = arr.searchsorted(data.EnclosedPorch)
    data.drop('EnclosedPorch', 1, inplace=True)
    arr = np.array([127, 127*2,127*3, 127*4])
    data['Sorted3SsnPorch'] = arr.searchsorted(data['3SsnPorch'])
    data.drop('3SsnPorch', 1, inplace=True)
    arr = np.array([120, 120*2,120*3, 120*4])
    data['SortedScreenPorch'] = arr.searchsorted(data.ScreenPorch)
    data.drop('ScreenPorch', 1, inplace=True)
    arr = np.array([3875, 3875*2,3875*3, 3875*4])
    data['SortedMiscVal'] = arr.searchsorted(data.MiscVal)
    data.drop('MiscVal', 1, inplace=True)
    arr = np.array([4,7,10,13])
    data['SortedMoSold'] = arr.searchsorted(data.MoSold)
    data.drop('MoSold', 1, inplace=True)
    arr = np.array([2007,2009,2011])
    data['SortedYrSold'] = arr.searchsorted(data.YrSold)
    data.drop('YrSold', 1, inplace=True)
    arr = np.array([50000, 100000, 15000, 200000, 250000, 300000, 350000, 400000, 450000, 500000,
                    550000, 600000, 650000, 700000, 750000, 800000])
    data['SortedPrice'] = arr.searchsorted(data.SalePrice)
    data.drop('SalePrice', 1, inplace=True)
    data.to_csv('./data/train_numbers_sorted.csv', index=False)
            
"""
splits the data set into a training data set that is 70% as big
"""
def getTrain(data):
    train=data.sample(frac=0.7,random_state=200)
    return train
"""
turns the othe 30% of the data into the test set
"""
def getTest(data, train):
    test = data.drop(train.index)
    return test

"""
calculate all the means for each attribute. This is possible becuase all discrete variables are
in numeric buckets.
"""

def attributeMeans(train):
    all_classes = set(list((data['SortedPrice'])))
    class_means = {}
    for clas in all_classes:
        means = []
        all_rows = train.loc[train['SortedPrice'] == clas]
        for column in all_rows:
            if column != 'SortedPrice':
                means.append(all_rows[column].mean())
        class_means[clas] = means
    return class_means

"""
calculate all the standarad deviations for each attribute. This is possible becuase all discrete 
variables are in numeric buckets.
"""
def attributeDeviations(train):
    all_classes = set(list((data['SortedPrice'])))
    class_deviations = {}
    for clas in all_classes:
        deviations = []
        all_rows = train.loc[train['SortedPrice'] == clas]
        for column in all_rows:
            deviations.append(all_rows[column].mean())
        class_deviations[clas] = deviations
    return class_deviations

"""
function to get probability density given input, mean and standard deviation
"""
def gaussianDensity(x, mean, stdev):
    try:
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
    except ZeroDivisionError:
        return 0
"""
function to calculate the probability that a test vector is in each class.
This returns a dictionary with a class bucket value as the key and the probability that the test
vector is in that class as its value
"""
def getClassProbabilities(means, deviations, inputline):
    classprobabilities = {}
    for (k,mean), (k2,deviation) in zip(means.items(), deviations.items()):
        classprobabilities[k] = 1
        for i in range(len(mean)):
            at_mean = mean[i]
            at_deviation = deviation[i]
            x = inputline[i]
            classprobabilities[k] += gaussianDensity(x, at_mean, at_deviation)
    return classprobabilities

"""
function returns class with the highest probability as found by the earlier method for a test vector.
Basically this function traverses the dictionary returned by getClassProbabilities and returns 
the class label for the class that has the hhighest probability. We could probably have built 
this into getClassProbabilities instead of making a seperate function
"""
            
def predictClass(means, deviations, inputline):
	probabilities = getClassProbabilities(means, deviations, inputline)
	label, prob = None, -1
	for k, v in probabilities.items():
		if label is None or v > prob:
			prob = v
			label = k
	return label

"""
reutrn the set of predictions for the entire test set instead of a single test vector
Basically itterates over each row in the test set and invokes the predictClass method on it
"""
 
def getPredictions(means, deviations, test):
    predictions = []
    for index,row in test.iterrows():
        testlist = list(row)
        result = predictClass(means, deviations, testlist)
        predictions.append(result)
    return predictions

"""
counts the number of correct predictions and then uses that to return the percentage correct
"""
 
def accuracy(predictions, testanswers):
	correct = 0
	for i in range(len(predictions)):
		if testanswers[i] == predictions[i]:
			correct += 1
	return (correct/float(len(testanswers))) * 100.0
 
if __name__ == "__main__":
    sortdata()
    train = getTrain(data)
    test = getTest(data, train)
    ##store answers for accuracy checking
    testanswers = list(test['SortedPrice'])
    ##delete answers from test set for accurate testing
    test.drop('SortedPrice', 1, inplace=True)
    means = attributeMeans(train)
    deviations = attributeDeviations(train)
    predictions = getPredictions(means, deviations, test)
    accuracy = accuracy(predictions, testanswers)
    print('Accuracy: ' + str(accuracy) + '%')