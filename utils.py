import math
import random
import re
import os

from sklearn.svm import SVC


def getSamples(filePath):
    """ Read data from data file.

    Parameters
    ----------
    filePath : A string, the path of the data file

    Returns
    -------
    A 2D List object, each row represents all gene expression values of a sample
    """
    file = open(filePath)
    firstLine = file.readline()
    # Get data size
    numberOfSamples = len(re.split(r'\t+', firstLine)) - 1
    numberOfGenes = open(filePath).read().count('\n') - 1
    # Init
    samples = [[None for x in range(numberOfGenes)] for y in range(numberOfSamples)]

    # Read data
    i = 0
    count = 0
    for line in file.readlines():
        geneExpression = line.split()
        # Ignore blank lines
        if len(geneExpression) == 0:
            continue
        # Each row of data:
        # The first value is the name of the gene (geneName = geneExpression[0]),
        # The second value starts with the expression value of the gene
        currentGeneExpression = geneExpression[1:]
        # Read the data in the file into an array
        j = 0
        for value in currentGeneExpression:
            samples[j][i] = 'None' if value == 'null' else float(value)
            j += 1
        i += 1
        count += 1
    file.close()
    return samples


def zScoreNormalization(samples):
    """Z-Score Normalization.

    Z-Score is used to normalize all gene expression data.

    Parameters
    ----------
    samples : 1 2D List, containing the gene expression values of all samples
    """
    averageValues = []
    stdVarianceValues = []

    if len(samples) == 0:
        raise Exception('No samples.')
    else:
        # Calculate the mean and standard deviation of each gene
        # (Only calculate the first row)
        numberOfGenes = len(samples[0])
        for index in range(0, numberOfGenes):
            average, stdVariance = getAverageAndVarianceAmongSamples(samples, index)
            averageValues.append(average)
            stdVarianceValues.append(stdVariance)

    # Z-Score Normalization
    for i, sample in enumerate(samples):
        for j, value in enumerate(sample):
            average = averageValues[j]
            stdVariance = stdVarianceValues[j]
            samples[i][j] = (value - average) / stdVariance

    return samples


def getAverageAndVarianceAmongSamples(samples, index):
    """Obtain the mean and standard deviation of the expression of a gene in the entire sample.

    And use the calculated mean to fill in missing values.

    Parameters
    ----------
    samples :  1 2D List, containing the gene expression values of all samples
    index : 1 int value, indicating the index value of the column to be calculated

    Returns
    -------
    2 double values, respectively representing the mean and standard deviation of a certain gene value
    """
    # The sum of all gene expression values
    sum = 0
    # Number of non-missing values
    numberOfExistingValue = 0
    # Sample mean and standard deviation
    average = 0
    stdVariance = 0

    # Calculate the sample mean
    for sample in samples:
        thisValue = sample[index]

        if thisValue != 'None':
            numberOfExistingValue += 1
            sum += thisValue
    average = sum / numberOfExistingValue

    # Calculate sample variance
    for dummyIndex, sample in enumerate(samples):
        thisValue = sample[index]

        if thisValue != 'None':
            stdVariance += (thisValue - average) * (thisValue - average)
        else:
            sample[index] = average
    stdVariance = math.sqrt(stdVariance / (numberOfExistingValue * 1.0))

    return average, stdVariance


def getTraningAndTestingSamples(samples, sampleIndexes, numberOfTraningSamples, numberOfTestingSamples):
    """Generate training data and test data from samples.

    Parameters
    ----------
    samples: 1 2D List, containing the gene expression values of all samples (dimension reduction)
    sampleIndexes: 1 1DList, indicating the sample index values of can be sampled
    numberOfTraningSamples: 1 int value, sampling as the number of training samples
    numberOfTestingSamples: 1 int value, sampled as the number of test samples

    Returns
    -------
    2 1D Lists, which store (dimension-reduced) training samples and (dimension-reduced) test samples respectively
    """
    trainingSampleIndexes = random.sample(sampleIndexes, numberOfTraningSamples)
    testingSampleIndexes = random.sample(list(set(sampleIndexes) - set(trainingSampleIndexes)), numberOfTestingSamples)

    trainingSamples = []
    testingSamples = []

    for index in trainingSampleIndexes:
        trainingSamples.append(samples[index])
    for index in testingSampleIndexes:
        testingSamples.append(samples[index])

    return trainingSamples, testingSamples


def getClassificationErrorSamples(trainingSamples, trainingSampleLabels, testingSamples, testingSampleLabels):
    """Evaluate the classification accuracy of the reduced data.

    Use the reduced data to train an SVM classifier and test the classification accuracy.

    Parameters
    ----------
    trainingSamples: A 2D List containing the gene expression values of training samples
    trainingSampleLabels: A 1D List, Class Labels containing training samples
    testingSamples: A 2D List containing the gene expression values of test samples
    testingSampleLabels: A 1D List containing Class Labels of test samples

    Returns
    -------
    5 int type data:
    -The first parameter indicates the number of misclassified samples on the training set
    -The second parameter represents TP (True Positive)
    -The third parameter represents FP (False Positive, Predicted Positive)
    -The fourth parameter represents FN (False Negative, Predicted Negative)
    -The fifth parameter means TN (False Negative)
    """
    # Create SVM
    clf = SVC()
    clf.fit(trainingSamples, trainingSampleLabels)
    predictedTrainingSampleLabels = clf.predict(trainingSamples)
    predictedTestingSampleLabels = clf.predict(testingSamples)

    # Compare Classification Result
    numberOfTrainingSamples = len(trainingSampleLabels)
    numberOfErrorTrainingSamples = 0
    numberOfTestingSamples = len(testingSampleLabels)
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    # Training Error Statistics
    for i in range(0, numberOfTrainingSamples):
        if trainingSampleLabels[i] != predictedTrainingSampleLabels[i]:
            numberOfErrorTrainingSamples = numberOfErrorTrainingSamples + 1

    # Testing Error Statistics
    # 0 stands for Normal, 1 stands for Tumor
    for i in range(0, numberOfTestingSamples):
        if testingSampleLabels[i] == 0 and predictedTestingSampleLabels[i] == 0:
            # TP
            tp += 1
        elif testingSampleLabels[i] == 1 and predictedTestingSampleLabels[i] == 0:
            # FP
            fp += 1
        elif testingSampleLabels[i] == 0 and predictedTestingSampleLabels[i] == 1:
            # FN
            fn += 1
        elif testingSampleLabels[i] == 1 and predictedTestingSampleLabels[i] == 1:
            # TN
            tn += 1

    return numberOfErrorTrainingSamples, tp, fp, fn, tn


def load_dataset(normalFilePath, tumorFilePath):
    # Sample of Normal People
    normalSamples = getSamples(normalFilePath)
    numberOfNormalSamples = len(normalSamples)

    # Samples of Tumor People
    tumorSamples = getSamples(tumorFilePath)
    numberOfTumorSamples = len(tumorSamples)

    # All Samples
    samples = normalSamples + tumorSamples
    numberOfSamples = numberOfNormalSamples + numberOfTumorSamples

    # Gene Indexes in List: samples
    normalSampleIndexes = range(0, numberOfNormalSamples)
    tumorSampleIndexes = range(numberOfNormalSamples, numberOfSamples)

    # numberOfGenes = len(samples[0]) if len(samples) != 0 else 0

    return samples, normalSampleIndexes, tumorSampleIndexes
