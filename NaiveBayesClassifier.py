'''
Created on Feb 18, 2014

@author: Karan K. Budhraja
'''

'''
Attribute Information: (x=player x has taken, o=player o has taken, b=blank)

    1. top-left-square: {x,o,b}
    2. top-middle-square: {x,o,b}
    3. top-right-square: {x,o,b}
    4. middle-left-square: {x,o,b}
    5. middle-middle-square: {x,o,b}
    6. middle-right-square: {x,o,b}
    7. bottom-left-square: {x,o,b}
    8. bottom-middle-square: {x,o,b}
    9. bottom-right-square: {x,o,b}
   10. Class: {positive,negative}
   
Distance Metric: number of changes in a feature value

    1. x vs o or vice versa -> 2
    2. x or o vs b or vice versa -> 1
'''

ATTRIBUTE_COUNT = 9
LABEL_INDEX = 9

def load_data():
    
    trainingSet = []
    testSet = []

    # number of instances of that class
    classInstances = {}
    classInstances["positive"] = 0
    classInstances["negative"] = 0

    # get data from file
    dataFile = open("tic-tac-toe.data", "r")

    for instanceData in dataFile:
        # divide the instances
        instance = instanceData.split(",")
        instance[LABEL_INDEX] = instance[LABEL_INDEX].strip()

        label = instance[LABEL_INDEX]
        classInstances[label] = classInstances[label] + 1
        
        if classInstances[label] == 3:
            # put it in the test set
            testSet.append(instance)
        else:
            # put it in the training set
            trainingSet.append(instance)
    
        # make sure the counter is less than 3
        classInstances[label] = classInstances[label] % 3
    
    return [trainingSet, testSet]

def train_classifier(trainingSet):

    # variables
    totalCount = {"positive" : 0, "negative" : 0}
    
    totalCountAttribute = {"positive":{"x" : 0, "o": 0, "b" : 0}, "negative":{"x" : 0, "o": 0, "b" : 0}}
    totalCountAttributes = [totalCountAttribute] * ATTRIBUTE_COUNT

    # go through all the instances
    for trainingInstance in trainingSet:
    
        label = trainingInstance[LABEL_INDEX]
    
        # label specific
        totalCount[label] = totalCount[label] + 1

        # attribute specific
        for attributeIndex in range(ATTRIBUTE_COUNT):
            totalCountAttributes[attributeIndex][label][trainingInstance[0]] = totalCountAttributes[attributeIndex][label][trainingInstance[0]] + 1
    
    # values to be calculated
    # specific to data set
    pLabel = {}
    pLabel["positive"] = totalCount["positive"]/len(trainingSet)
    pLabel["negative"] = totalCount["negative"]/len(trainingSet)

    pFeature = {"positive":{"x" : 0, "o": 0, "b" : 0}, "negative":{"x" : 0, "o": 0, "b" : 0}}   
    pFeatures = [pFeature] * ATTRIBUTE_COUNT

    for attributeIndex in range(ATTRIBUTE_COUNT):        
        
        pFeatures[attributeIndex]["positive"]["x"] = totalCountAttributes[attributeIndex]["positive"]["x"] / (totalCountAttributes[attributeIndex]["positive"]["x"] + totalCountAttributes[attributeIndex]["negative"]["x"])
        pFeatures[attributeIndex]["positive"]["o"] = totalCountAttributes[attributeIndex]["positive"]["o"] / (totalCountAttributes[attributeIndex]["positive"]["o"] + totalCountAttributes[attributeIndex]["negative"]["o"])
        pFeatures[attributeIndex]["positive"]["b"] = totalCountAttributes[attributeIndex]["positive"]["b"] / (totalCountAttributes[attributeIndex]["positive"]["b"] + totalCountAttributes[attributeIndex]["negative"]["b"])
        pFeatures[attributeIndex]["negative"]["x"] = 1 - pFeatures[attributeIndex]["positive"]["x"]
        pFeatures[attributeIndex]["negative"]["o"] = 1 - pFeatures[attributeIndex]["positive"]["o"]
        pFeatures[attributeIndex]["negative"]["b"] = 1 - pFeatures[attributeIndex]["positive"]["b"]
                
    return [pLabel, pFeatures]

def test_classifier(testSet, pLabel, pFeatures):
    
    # count correctly and incorrectly classified instances
    correctlyLabeled = 0
    incorrectlyLabeled = 0
    
    # check each instance
    for testInstance in testSet:
        
        pPositiveGivenX = 1
        pNegativeGivenX = 1
        
        # calculate product of attribute based probabilities
        for attributeIndex in range(ATTRIBUTE_COUNT):
            pPositiveGivenX = pPositiveGivenX * pFeatures[attributeIndex]["positive"][testInstance[attributeIndex]]
            pNegativeGivenX = pNegativeGivenX * pFeatures[attributeIndex]["negative"][testInstance[attributeIndex]]
        
        # we will ignore a constant P(X) for both positive and negative
        pPositiveFactor = pLabel["positive"] * pPositiveGivenX
        pNegativeFactor = pLabel["negative"] * pNegativeGivenX
        
        if pPositiveFactor > pNegativeFactor:
            predictedLabel = "positive"
        else:
            predictedLabel = "negative"
    
        # check if label is correct
        if predictedLabel == testInstance[LABEL_INDEX]:
            correctlyLabeled = correctlyLabeled + 1
        else:
            incorrectlyLabeled = incorrectlyLabeled + 1
    
    # calculate accuracy
    accuracy = correctlyLabeled / (correctlyLabeled + incorrectlyLabeled)

    return accuracy

def main():            

    # load data from file
    [trainingSet, testSet] = load_data()
    
    # train based on the training set
    [pLabel, pFeatures] = train_classifier(trainingSet)
    
    # test the classifier using training data
    accuracy = test_classifier(testSet, pLabel, pFeatures)
    print("Classification Accuracy = " + str(accuracy))
    
main()