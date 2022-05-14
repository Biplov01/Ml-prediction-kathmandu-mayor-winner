
import pandas as pd
import numpy as np
import time

#Data with features and target values
#Tutorial for Pandas is here - https://pandas.pydata.org/pandas-docs/stable/tutorials.html
#Helper functions are provided so you shouldn't need to learn Pandas
dataset = pd.read_csv("data.csv")

#========================================== Data Helper Functions ==========================================

#Normalize values between 0 and 1
#dataset: Pandas dataframe
#categories: list of columns to normalize, e.g. ["column A", "column C"]
#Return: full dataset with normalized values
def normalizeData(dataset, categories):
    normData = dataset.copy()
    col = dataset[categories]
    col_norm = (col - col.min()) / (col.max() - col.min())
    normData[categories] = col_norm
    return normData

#Encode categorical values as mutliple columns (One Hot Encoding)
#dataset: Pandas dataframe
#categories: list of columns to encode, e.g. ["column A", "column C"]
#Return: full dataset with categorical columns replaced with 1 column per category
def encodeData(dataset, categories):
	return pd.get_dummies(dataset, columns=categories)

#Split data between training and testing data
#dataset: Pandas dataframe
#ratio: number [0, 1] that determines percentage of data used for training
#Return: (Training Data, Testing Data)
def trainingTestData(dataset, ratio):
	tr = int(len(dataset)*ratio)
	return dataset[:tr], dataset[tr:]

#Convenience function to extract Numpy data from dataset
#dataset: Pandas dataframe
#Return: features numpy array and corresponding labels as numpy array
def getNumpy(dataset):
	features = dataset.drop(["can_id", "can_nam","winner"], axis=1).values
	labels = dataset["winner"].astype(int).values
	return features, labels

#Convenience function to extract data from dataset (if you prefer not to use Numpy)
#dataset: Pandas dataframe
#Return: features list and corresponding labels as a list
def getPythonList(dataset):
	f, l = getNumpy(dataset)
	return f.tolist(), l.tolist()

#Calculates accuracy of your models output.
#solutions: model predictions as a list or numpy array
#real: model labels as a list or numpy array
#Return: number between 0 and 1 representing your model's accuracy
def evaluate(solutions, real):
	predictions = np.array(solutions)
	labels = np.array(real)
	return float((predictions == labels).sum()) / labels.size

#===========================================================================================================
##Names of columns to be normalized and having categorical data for encoding
##Changes names here if data have different column names
normalizeColoumns = ["net_ope_exp", "net_con", "tot_loa"]
catogoricalColoumns = ['can_off', 'can_inc_cha_ope_sea']

## Preprocess function for KNN
def preprocessKNN(dataSet):
	normalizedDataset = normalizeData(dataSet, normalizeColoumns)
	encodedDataset = encodeData(normalizedDataset, catogoricalColoumns)
	f, l = getPythonList(encodedDataset)
	return (f,l)

## Preprocess function for Perceptron and MLP
def preprocess(dataSet):
	normalizedDataset = normalizeData(dataSet, normalizeColoumns)
	encodedDataset = encodeData(normalizedDataset, catogoricalColoumns)
	encodedDataset['bais'] = pd.Series(1, index=encodedDataset.index)
	f, l = getPythonList(encodedDataset)
	return (f,l)

## Preprocess function for ID3
def preprocessID3(dataSet):
	newDataset = dataSet.drop(["can_id", "can_nam"], axis=1)
	normalizedDataset = normalizeData(newDataset, normalizeColoumns)
	normalizedDataset["winner"] = normalizedDataset["winner"].astype(int).values
	labels = normalizedDataset["winner"].astype(int).values.tolist()
	return (normalizedDataset, labels)


class KNN:

	def __init__(self):
		#KNN state here
		#Feel free to add methods
		pass

	def train(self, features, labels):
		#training logic here
		#input is list/array of features and labels
		self.kValue = 6
		self.trainingFeatures = features
		self.trainingLabels = labels

	def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features
		predictedResults = []
		for i in range(len(features)):
			distanceArray = []
			for j in range(len(self.trainingFeatures)):
				distance = self.getEuclideanDistance(features[i], self.trainingFeatures[j])
				distanceArray.append((distance, self.trainingLabels[j]))
			distanceArray = sorted(distanceArray, key=lambda dist: dist[0])
			predictedResults.append(self.getHighestVotes(distanceArray))

		return predictedResults

	def getEuclideanDistance(self, feature1, feature2):
		dist = 0
		for i in range(len(feature2)):
			dist+= pow((feature2[i] - feature1[i]),2)
		return np.sqrt(dist)

	def getHighestVotes(self, list):
		votes = {}
		for k in range(self.kValue):
			if list[k][1] in votes:
				votes[list[k][1]] += 1
			else:
				votes[list[k][1]] = 1
		return max(votes, key=votes.get)


class Perceptron:
	def __init__(self):
		#Perceptron state here
		#Feel free to add methods
		pass


	def train(self, features, labels):
		#training logic here
		#input is list/array of features and labels
		self.learningRate = 0.01
		self.weights = [np.random.uniform(-1.0,1.0) for i in range(len(features[0]))]
		bais = 0.1
		self.weights.append(bais)
		oldWeights = []
		timeout = time.time()+60
		epochCount = 0
		while (time.time() <= timeout):
			epochCount += 1
			for i in range(len(features)):
				predictedValue = self.predictTraining(features[i])
				diff  = labels[i] - predictedValue
				if diff != 0:
					if labels[i] == 0:
						diff = -1
					else:
						diff = 1
				for j in range(len(features[i])):
					self.weights[j] = round((self.weights[j] + self.learningRate * diff * features[i][j]),3)
			##End training before 1 minute if 6 full iterations result in no change in weights.(Breaks the while loop)
			if len(oldWeights) == 6:
				oldWeights.pop(0)
				oldWeights.append(self.weights[:])
				if oldWeights[1:] == oldWeights[:-1]:
					break
			else:
				oldWeights.append(self.weights[:])


	def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features
		activationOutputs = []
		for i in range(len(features)):
			summation = 0
			for j in range(len(features[i])):
				summation+=(features[i][j]*self.weights[j])
			activationOutputs.append(round(self.sigmoid(summation)))
		return activationOutputs

	def predictTraining(self, feature):
		summation = 0
		for i in range(len(feature)):
			summation += (feature[i] * self.weights[i])
		activationOutputs = (self.stepFunction(summation))
		return activationOutputs

	def sigmoid(self, x):
		return 1/(1 + np.exp(-x))

	def stepFunction(self, x):
		if x>0:
			return 1
		else:
			return 0


class MLP:
	def __init__(self):
		#Multilayer perceptron state here
		#Feel free to add methods
		pass


	def train(self, features, labels):
		#training logic here
		#input is list/array of features and labels
		self.learningRate = 0.01
		self.hiddenLayer = 1
		self.hiddenNodes = len(features[0])
		self.allWeights = []
		self.initializeAllWeights(features)
		oldWeights = []
		timeout = time.time() + 60
		layers = self.hiddenLayer + 1
		while (time.time() <= timeout):
			for i in range(len(features)):
				input = np.matrix(features[i]).copy()
				sigmoidOutput = []
				sigmoidOutput.append(input)
				## Forward model propogation
				for j in range(layers):
					newOutput = self.forwardOutput(input, self.allWeights[j])
					output=[]
					for x in newOutput.tolist():
						for y in x:
							output.append(self.sigmoid(y))
					sigmoidOutput.append(np.matrix(output))
					input = np.matrix(output).copy()
				## Backpropogation Model
				deltaModel = []
				for dummy in range(layers):
					deltaModel.append([])
				delta = -(sigmoidOutput[-1].item())*(1-sigmoidOutput[-1].item())*(labels[i] - sigmoidOutput[-1].item())
				deltaModel[-1] = np.matrix(delta)
				for k in range(layers-2,-1,-1):
					sigValue = self.subtractSigma(sigmoidOutput[k+1])
					derivativeSig = np.multiply(sigmoidOutput[k+1], sigValue)
					delta = deltaModel[k+1].dot(self.allWeights[k+1])
					deltaModel[k] = np.multiply(derivativeSig,delta)
				for w in range(len(self.allWeights)):
					self.allWeights[w] = self.allWeights[w] - np.multiply(self.learningRate,np.multiply(sigmoidOutput[w],deltaModel[w]))
			##End training before 1 minute if 6 full iterations result in no change in weights.(Breaks the while loop)
			if len(oldWeights) == 6:
				oldWeights.pop(0)
				oldWeights.append(self.allWeights[:])
				if oldWeights == oldWeights.reverse():
					break
			else:
				oldWeights.append(self.allWeights[:])


	def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features
		predictedValues = []
		for i in range(len(features)):
			layers = self.hiddenLayer + 1
			input = np.matrix(features[i]).copy()
			## Forward model propogation
			for j in range(layers):
				newOutput = self.forwardOutput(input, self.allWeights[j])
				output = []
				for x in newOutput.tolist():
					for y in x:
						output.append(self.sigmoid(y))
				input = np.matrix(output).copy()
			predictedValues.append(round(input.item()))
		return predictedValues

	def initializeAllWeights(self, features):
		self.allWeights.append(np.random.uniform(-1.0, 1.0, (self.hiddenNodes ,len(features[0]))))
		if self.hiddenLayer > 1:
			for i in range(1,self.hiddenLayer):
				self.allWeights.append(np.random.uniform(-1.0, 1.0, (self.hiddenNodes, self.hiddenNodes)))
		self.allWeights.append(np.random.uniform(-1.0, 1.0, (1, self.hiddenNodes)))

	def forwardOutput(self, feature, weights):
		return np.dot(feature, np.transpose(weights))

	def sigmoid(self, x):
		return 1/(1 + np.exp(-x))

	def updateWeight(self, weight, learningRate, derivative):
		return weight + learningRate*derivative

	def subtractSigma(self, matrix):
		output = []
		for x in matrix.tolist():
			for y in x:
				output.append(1-y)
		return np.matrix(output)



class ID3:
	def __init__(self):
		#Decision tree state here
		#Feel free to add methods
		pass

	def train(self, features, labels):
		#training logic here
		#input is list/array of features and labels
		self.mostOccuringOutput = max(labels, key=labels.count)
		colNames = list(features)
		colNames.remove("jitne")
		tree = self.createNewTree(features, colNames, "jitne", 0)
		self.tree = tree

	def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features
		predictions = []
		for (idx,row) in features.iterrows():
			updatedTree = self.tree.copy()
			updatedRow = row.copy()
			while updatedTree != 1 or updatedTree != 0:
				node = updatedTree.keys()[0]
				branchValue = updatedRow.loc[node]
				valueCopy = branchValue
				ranges = updatedTree[node].keys()
				## If node is an interval of numbers,
				## set branch as interval bin which contains the sample value
				if not isinstance(branchValue, str):
					diffValue = 999999
					for j in range(len(ranges)):
						l,h = ranges[j]
						if l <= valueCopy <= h:
							branchValue = ranges[j]
							break
						else:
							mean = (l+h)/2
							diff = np.sqrt((valueCopy-mean)**2)
							if diff <= diffValue:
								diffValue = diff
								branchValue = ranges[j]
				if branchValue not in ranges:
					updatedTree = np.random.randint(2)
					predictions.append(updatedTree)
					break
				## If node value is 0 or 1
				if isinstance(updatedTree[node][branchValue], int):
					updatedTree = updatedTree[node][branchValue]
				## if node value gives a dictionary
				else:
					updatedTree = updatedTree[node][branchValue].copy()
				## If node value is 0 or 1, end tree search and return value
				if updatedTree == 0:
					predictions.append(0)
					break
				elif updatedTree == 1:
					predictions.append(1)
					break
		return predictions

	## Return attribute and bin with highest gain
	def findBestAttribute(self, featureSet, featureNames, targetFeature):
		features = featureSet.copy()
		highestGain = 0
		finalColoumn = featureNames[0]
		bins = []
		for attribute in featureNames:
			gain, binList = self.findGain(features, featureNames, attribute, targetFeature)
			if gain>=highestGain:
				highestGain = gain
				finalColoumn = attribute
				bins = binList
		return (finalColoumn, bins)

	## Return gain and bins of each attribute in a dataset
	def findGain(self, featureSet, featureNames, attribute, targetFeature):
		features = featureSet.copy()
		attrInd = featureNames.index(attribute)
		labels = features[targetFeature].astype(int).values.tolist()
		splitArrtibute = []
		binList = []
		occurunce = {}
		## For categorical data of attribute
		if features[attribute].dtype is np.dtype(object):
			for row in features.values:
				if row[attrInd] not in binList:
					binList.append(row[attrInd])
			splitArrtibute = features[attribute].values.tolist()
		## For numerical data of attribute
		else:
			for row in features.values:
				if (occurunce.has_key(row[attrInd])):
					occurunce[row[attrInd]]+=1.0
				else:
					occurunce[row[attrInd]]=1.0
			if len(occurunce) >= 5:
				splitArrtibute, binList = pd.cut(features[attribute], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], retbins=True, include_lowest=True)
			else:
				bins = len(occurunce)
				splitArrtibute, binList = pd.cut(features[attribute], bins, retbins=True)
			splitArrtibute = splitArrtibute.values.tolist()
		splitAttributeData = {}

		## Get dictionary of 1s and 0s for each bin of the attribute values
		for i in range(len(splitArrtibute)):
			if (labels[i] == 1):
				if splitArrtibute[i] in splitAttributeData:
					splitAttributeData[splitArrtibute[i]][1]+=1
				else:
					splitAttributeData[splitArrtibute[i]] = [0,1]
			else:
				if splitArrtibute[i] in splitAttributeData:
					splitAttributeData[splitArrtibute[i]][0] += 1
				else:
					splitAttributeData[splitArrtibute[i]] = [1, 0]

		gain = self.getEntropyValues(features, splitAttributeData)
		return (gain,binList)

	##Calculate entropy values and return information gain
	def getEntropyValues(self, featureSet, data):
		features = featureSet.copy()
		totalSamples = float(len(features.values))
		attInfoGain = 0.0
		totalPostiveSamples = 0
		totalNegativeSamples = 0
		for i in data.keys():
			totalPostiveSamples+=data.get(i)[1]
			totalNegativeSamples+=data.get(i)[0]
			attInfoGain+=(data.get(i)[1]+data.get(i)[0])/totalSamples*self.entropyFormula(data.get(i)[1],data.get(i)[0])

		infoGain = self.entropyFormula(totalPostiveSamples, totalNegativeSamples)
		return infoGain-attInfoGain

	def entropyFormula(self,p,n):
		p=float(p)
		n=float(n)
		if p == 0.0:
			return -(n/(p+n))*np.log2(n/(n+p))
		elif n ==0.0:
			return -(p / (p + n)) * np.log2(p / (p + n))
		else:
			return - (p/(p+n))*np.log2(p/(p+n)) - (n/(p+n))*np.log2(n/(n+p))

	##Function that returns a Decision Tree
	def createNewTree(self, features, featureNames, targetAttribute, oldFeatures):
		features=features.copy()
		labels = features[targetAttribute].astype(int).values.tolist()

		##If dataset is empty (no data for chosen attribute and bin),
		## then return most occuring target value of the parent attribute
		if features.empty or not featureNames:
			oldLabels = oldFeatures[targetAttribute].astype(int).values.tolist()
			return max(oldLabels, key=oldLabels.count)

		## If all remaining target values are 0 or all 1, then return the target
		elif len(labels) == labels.count(labels[0]):
			return labels[0]

		##All other cases make a new subtree
		else:
			bestAttr, binList = self.findBestAttribute(features, featureNames, targetAttribute)
			decisionTree = {bestAttr:{}}

			## If bins are numerical
			if isinstance(binList[0], float):
				binLength = len(binList)-1
				for i in range(binLength):
					rangeTuple = (binList[i],binList[i+1])
					## Get dataset for current bin
					filteredFeatures = self.getNumFilteredDataset(features, bestAttr, rangeTuple)
					filteredFeatureNames = list(filteredFeatures)
					filteredFeatureNames.remove(targetAttribute)
					subDecisionTree = self.createNewTree(filteredFeatures, filteredFeatureNames,targetAttribute, features)
					decisionTree[bestAttr][rangeTuple] = subDecisionTree
			## If bins are categorical data
			elif isinstance(binList[0], str):
				binLength = len(binList)
				for i in range(binLength):
					## Get dataset for current bin
					filteredFeatures = self.getCatFilteredDataset(features, bestAttr, binList[i])
					filteredFeatureNames = list(filteredFeatures)
					filteredFeatureNames.remove(targetAttribute)
					subDecisionTree = self.createNewTree(filteredFeatures, filteredFeatureNames,targetAttribute, features)
					decisionTree[bestAttr][binList[i]] = subDecisionTree

		return  decisionTree

	##Get filtered dataset based on best attribute chosen and numerical branch(bin) range value
	def getNumFilteredDataset(self, features, bestAttr, range):
		newFeatures = features.loc[features[bestAttr].between(range[0], range[1]+0.001, inclusive=False)]
		newFeatures = newFeatures.drop([bestAttr], axis=1)
		return newFeatures

	##Get filtered dataset based on best attribute chosen and categorical branch(bin) value
	def getCatFilteredDataset(self, features, bestAttr, value):
		newFeatures = features.loc[features[bestAttr] == value]
		newFeatures = newFeatures.drop([bestAttr], axis=1)
		return newFeatures


knnModel = KNN()
trainSet, testSet = trainingTestData(dataset, 0.7)
fTrain, lTrain = preprocessKNN(trainSet)
knnModel.train(fTrain, lTrain)
fTest, lTest = preprocessKNN(testSet)
predictedOutputs = knnModel.predict(fTest)
accuracy = evaluate(predictedOutputs, lTest)
print ("KNN Accuracy : "+str(accuracy*100))


perceptronModel = Perceptron()
trainSet, testSet = trainingTestData(dataset, 0.8)
fTrain, lTrain = preprocess(trainSet)
perceptronModel.train(fTrain, lTrain)
fTest, lTest = preprocess(testSet)
predictedOutputs = perceptronModel.predict(fTest)
accuracy = evaluate(predictedOutputs, lTest)
print ("Perceptron Accuracy : "+str(accuracy*100))


mlpModel = MLP()
trainSet, testSet = trainingTestData(dataset, 0.8)
fTrain, lTrain = preprocess(trainSet)
mlpModel.train(fTrain, lTrain)
fTest, lTest = preprocess(testSet)
predictedOutputs = mlpModel.predict(fTest)
accuracy = evaluate(predictedOutputs, lTest)
print ("MLP Accuracy : "+str(accuracy*100))


id3Model = ID3()
trainSet, testSet = trainingTestData(dataset, 0.8)
fTrain, lTrain = preprocessID3(trainSet)
id3Model.train(fTrain, lTrain)
fTest, lTest = preprocessID3(testSet)
predictedOutputs = id3Model.predict(fTest)
accuracy = evaluate(predictedOutputs, lTest)
print ("ID3 Accuracy : "+str(accuracy*100))
