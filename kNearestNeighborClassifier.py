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

def calculate_distance(instance1, instance2):
    
    attributeDistances = []
    
    # loop over all attributes
    for attributeIndex in range(ATTRIBUTE_COUNT):
        
        instance1Attribute = instance1[attributeIndex]
        instance2Attribute = instance2[attributeIndex]
        
        # determine distance
        if instance1Attribute == "x" and instance2Attribute == "o":
            attributeDistance = 2
        elif instance1Attribute == "o" and instance2Attribute == "x":
            attributeDistance = 2
            
        elif instance1Attribute != instance2Attribute:
            # one of them is blank
            attributeDistance = 1
        else:
            # attributes have the same value
            attributeDistance = 0
    
        attributeDistances.append(attributeDistance)
            
    # euclidean distance
    squareAttributeDistances = [attributeDistance*attributeDistance for attributeDistance in attributeDistances]    
    return sum(squareAttributeDistances)
            
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

def test_classifier(k, trainingSet, testSet):
    
    # count correctly and incorrectly classified instances
    correctlyLabeled = 0
    incorrectlyLabeled = 0
    
    # check each instance
    for testInstance in testSet:
        
        # store tuples as [distance, trainingInstance]
        INSTANCE_INDEX = 1
        distances = []
        
        # calculate distance with respect to all instances
        for trainingInstance in trainingSet:
                        
            # if we are testing with trainingSet, do not compare with itself
            if testInstance == trainingInstance:
                continue
                        
            # calculate distance
            attributeDistance = calculate_distance(testInstance, trainingInstance)            
            distances.append([attributeDistance, trainingInstance])
            
        # sort distances in ascending order
        distances.sort()
        
        # get the k nearest neighbors
        neighbors = distances[:k]
        
        # now count positive and negative neighbors
        positiveNeighbors = 0
        negativeNeighbors = 0
        
        for neighbor in neighbors:
            
            if neighbor[INSTANCE_INDEX][LABEL_INDEX] == "positive":
                positiveNeighbors = positiveNeighbors + 1
            else:
                negativeNeighbors = negativeNeighbors + 1
            
        # calculate label based on majority
        if (positiveNeighbors > negativeNeighbors):
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

    # define k
    k = 5
        
    # test the classifier on training data
    accuracy = test_classifier(k, trainingSet, trainingSet)
    print("Classification Accuracy on Training Set= " + str(accuracy))

    # test the classifier on test data
    accuracy = test_classifier(k, trainingSet, testSet)
    print("Classification Accuracy on Test Set= " + str(accuracy))
    
main()