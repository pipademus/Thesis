import pandas as pd
import datatest
from sklearn import tree, metrics, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# variables containing the datasets columns
columnsGeneral = {'callType', 'startTime', 'elapsedTime', 'success', 'traceId', 'id', 'pid', 'cmdb_id', 'serviceName'}
columnsJDBC = {'callType', 'startTime', 'elapsedTime', 'success', 'traceId', 'id', 'pid', 'cmdb_id', 'dsName'}
columnsLocal = {'callType', 'startTime', 'elapsedTime', 'success', 'traceId', 'id', 'pid', 'cmdb_id', 'serviceName', 'dsName'}

# function to change the categorical data to numerical data
def categoricalToNumerical (metrics, columns):
    # assign the 'categorical' type to the dataframe columns
    for i in columns:
        # verify if the column isn't the hyperparameter
        if i != 'success':
            # verify if the column isn't one of the numericals
            if i != 'startTime' or i != 'elapsedTime':
                metrics[i] = metrics[i].astype('category')

    # select the columns from the set
    categoricalColumns = metrics.select_dtypes(['category']).columns
    # transform the categorical data into numerical data
    metrics[categoricalColumns] = metrics[categoricalColumns].apply(lambda x: x.cat.codes)

# open datasets
dataframeCSF = pd.read_csv('trace_csf.csv', sep = ',')
dataframeFR = pd.read_csv('trace_fly_remote.csv', sep = ',')
dataframeJDBC = pd.read_csv('trace_jdbc.csv', sep = ',')
dataframeLocal = pd.read_csv('trace_local.csv', sep = ',')
dataframeOSB = pd.read_csv('trace_osb.csv', sep = ',')
dataframeRP = pd.read_csv('trace_remote_process.csv', sep = ',')

# verify if the data includes the expected column names
datatest.validate(dataframeCSF.columns, columnsGeneral)
datatest.validate(dataframeFR.columns, columnsGeneral)
datatest.validate(dataframeJDBC.columns, columnsJDBC)
datatest.validate(dataframeLocal.columns, columnsLocal)
datatest.validate(dataframeOSB.columns, columnsGeneral)
datatest.validate(dataframeRP.columns, columnsGeneral)

# ----- TRACE-CSF ----- #
# get 80% of the dataset using a seed to randomize the items for training
trainSetCSF = dataframeCSF.sample(frac = 0.8, random_state = 200)
# drop the selected items for the training and keep the remaining items for testing
testSetCSF = dataframeCSF.drop(trainSetCSF.index)
# get the testing result column
testResultCSF = testSetCSF.pop('success')
# keep the remaining columns for testing
testMetricsCSF = testSetCSF
# remove the result column to train what is to be expected
trainResultCSF = trainSetCSF.pop('success')
# keep the remaining columns for the training meters
trainMetricsCSF = trainSetCSF

# change the categorical columns of the training and testing meter sets to numerical
categoricalToNumerical(trainMetricsCSF, columnsGeneral)
categoricalToNumerical(testMetricsCSF, columnsGeneral)

print("trace_csf")

# create a decision tree classifier
treeCSF = tree.DecisionTreeClassifier()
# use the training metrics and results on the tree
treeCSF = treeCSF.fit(trainMetricsCSF, trainResultCSF)
# predict a result using the test metrics
predictTreeCSF = treeCSF.predict(testMetricsCSF)
# verify the accuracy between the testing results and the predicted values
print("Accuracy DTC: ", metrics.accuracy_score(testResultCSF, predictTreeCSF))

# create a support vector machine
svmCSF = svm.SVC(C = 1)
# use the training metrics and results on the machine
svmCSF.fit(trainMetricsCSF, trainResultCSF)
# predict a result using the test metrics
predictSvmCSF = svmCSF.predict(testMetricsCSF)
# verify the accuracy between the testing results and the predicted values
print("Accuracy SVM: ", metrics.accuracy_score(testResultCSF, predictSvmCSF))

# create a gaussian naive bayes
gnbCSF = GaussianNB()
# use the training metrics and results to predict a result
predictGnbCSF = gnbCSF.fit(trainMetricsCSF, trainResultCSF).predict(testMetricsCSF)
# verify the accuracy between the testing results and the predicted values
print("Accuracy GNB: ", metrics.accuracy_score(testResultCSF, predictGnbCSF))

# create a k neighbors classifier
kncCSF = KNeighborsClassifier(n_neighbors = 10)
# use the training metrics and results on the classifier
kncCSF.fit(trainMetricsCSF, trainResultCSF)
# predict a result using the test metrics
predictKncCSF = kncCSF.predict(testMetricsCSF)
# verify the accuracy between the testing results and the predicted values
print("Accuracy KNC: ", metrics.accuracy_score(testResultCSF, predictKncCSF))

print()

# ----- TRACE-FR ----- #
# get 80% of the dataset using a seed to randomize the items for training
trainSetFR = dataframeFR.sample(frac = 0.8, random_state = 200)
# drop the selected items for the training and keep the remaining items for testing
testSetFR = dataframeFR.drop(trainSetFR.index)
# get the testing result column
testResultFR = testSetFR.pop('success')
# keep the remaining columns for testing
testMetricsFR = testSetFR
# remove the result column to train what is to be expected
trainResultFR = trainSetFR.pop('success')
# keep the remaining columns for the training meters
trainMetricsFR = trainSetFR

# change the categorical columns of the training and testing meter sets to numerical
categoricalToNumerical(trainMetricsFR, columnsGeneral)
categoricalToNumerical(testMetricsFR, columnsGeneral)

print("trace_fly_remote")

# create a decision tree classifier
treeFR = tree.DecisionTreeClassifier()
# use the training metrics and results on the tree
treeFR = treeFR.fit(trainMetricsFR, trainResultFR)
# predict a result using the test metrics
predictTreeFR = treeFR.predict(testMetricsFR)
# verify the accuracy between the testing results and the predicted values
print("Accuracy DTC: ", metrics.accuracy_score(testResultFR, predictTreeFR))

# create a gaussian naive bayes
gnbFR = GaussianNB()
# use the training metrics and results to predict a result
predictGnbFR = gnbFR.fit(trainMetricsFR, trainResultFR).predict(testMetricsFR)
# verify the accuracy between the testing results and the predicted values
print("Accuracy GNB: ", metrics.accuracy_score(testResultFR, predictGnbFR))

# create a k neighbors classifier
kncFR = KNeighborsClassifier(n_neighbors = 10)
# use the training metrics and results on the classifier
kncFR.fit(trainMetricsFR, trainResultFR)
# predict a result using the test metrics
predictKncFR = kncFR.predict(testMetricsFR)
# verify the accuracy between the testing results and the predicted values
print("Accuracy KNC: ", metrics.accuracy_score(testResultFR, predictKncFR))

print()

# ----- TRACE-JDBC ----- #
# get 80% of the dataset using a seed to randomize the items for training
trainSetJDBC = dataframeJDBC.sample(frac = 0.8, random_state = 200)
# drop the selected items for the training and keep the remaining items for testing
testSetJDBC = dataframeJDBC.drop(trainSetJDBC.index)
# get the testing result column
testResultJDBC = testSetJDBC.pop('success')
# keep the remaining columns for testing
testMetricsJDBC = testSetJDBC
# remove the result column to train what is to be expected
trainResultJDBC = trainSetJDBC.pop('success')
# keep the remaining columns for the training meters
trainMetricsJDBC = trainSetJDBC

# change the categorical columns of the training and testing meter sets to numerical
categoricalToNumerical(trainMetricsJDBC, columnsJDBC)
categoricalToNumerical(testMetricsJDBC, columnsJDBC)

print("trace_jdbc")

# create a decision tree classifier
treeJDBC = tree.DecisionTreeClassifier()
# use the training metrics and results on the tree
treeJDBC = treeJDBC.fit(trainMetricsJDBC, trainResultJDBC)
# predict a result using the test metrics
predictTreeJDBC = treeJDBC.predict(testMetricsJDBC)
# verify the accuracy between the testing results and the predicted values
print("Accuracy DTC: ", metrics.accuracy_score(testResultJDBC, predictTreeJDBC))

# create a support vector machine
svmJDBC = svm.SVC(C = 1)
# use the training metrics and results on the machine
svmJDBC.fit(trainMetricsJDBC, trainResultJDBC)
# predict a result using the test metrics
predictSvmJDBC = svmJDBC.predict(testMetricsJDBC)
# verify the accuracy between the testing results and the predicted values
print("Accuracy SVM: ", metrics.accuracy_score(testResultJDBC, predictSvmJDBC))

# create a gaussian naive bayes
gnbJDBC = GaussianNB()
# use the training metrics and results to predict a result
predictGnbJDBC = gnbJDBC.fit(trainMetricsJDBC, trainResultJDBC).predict(testMetricsJDBC)
# verify the accuracy between the testing results and the predicted values
print("Accuracy GNB: ", metrics.accuracy_score(testResultJDBC, predictGnbJDBC))

# create a k neighbors classifier
kncJDBC = KNeighborsClassifier(n_neighbors = 10)
# use the training metrics and results on the classifier
kncJDBC.fit(trainMetricsJDBC, trainResultJDBC)
# predict a result using the test metrics
predictKncJDBC = kncJDBC.predict(testMetricsJDBC)
# verify the accuracy between the testing results and the predicted values
print("Accuracy KNC: ", metrics.accuracy_score(testResultJDBC, predictKncJDBC))

print()

# ----- TRACE-LOCAL ----- #
# get 80% of the dataset using a seed to randomize the items for training
trainSetLOCAL = dataframeLocal.sample(frac = 0.8, random_state = 200)
# drop the selected items for the training and keep the remaining items for testing
testSetLOCAL = dataframeLocal.drop(trainSetLOCAL.index)
# get the testing result column
testResultLOCAL = testSetLOCAL.pop('success')
# keep the remaining columns for testing
testMetricsLOCAL = testSetLOCAL
# remove the result column to train what is to be expected
trainResultLOCAL = trainSetLOCAL.pop('success')
# keep the remaining columns for the training meters
trainMetricsLOCAL = trainSetLOCAL

# change the categorical columns of the training and testing meter sets to numerical
categoricalToNumerical(trainMetricsLOCAL, columnsLocal)
categoricalToNumerical(testMetricsLOCAL, columnsLocal)

print("trace_local")

# create a decision tree classifier
treeLOCAL = tree.DecisionTreeClassifier()
# use the training metrics and results on the tree
treeLOCAL = treeLOCAL.fit(trainMetricsLOCAL, trainResultLOCAL)
# predict a result using the test metrics
predictTreeLOCAL = treeLOCAL.predict(testMetricsLOCAL)
# verify the accuracy between the testing results and the predicted values
print("Accuracy DTC: ", metrics.accuracy_score(testResultLOCAL, predictTreeLOCAL))

# create a support vector machine
svmLOCAL = svm.SVC(C = 1)
# use the training metrics and results on the machine
svmLOCAL.fit(trainMetricsLOCAL, trainResultLOCAL)
# predict a result using the test metrics
predictSvmLOCAL = svmLOCAL.predict(testMetricsLOCAL)
# verify the accuracy between the testing results and the predicted values
print("Accuracy SVM: ", metrics.accuracy_score(testResultLOCAL, predictSvmLOCAL))

# create a gaussian naive bayes
gnbLOCAL = GaussianNB()
# use the training metrics and results to predict a result
predictGnbLOCAL = gnbLOCAL.fit(trainMetricsLOCAL, trainResultLOCAL).predict(testMetricsLOCAL)
# verify the accuracy between the testing results and the predicted values
print("Accuracy GNB: ", metrics.accuracy_score(testResultLOCAL, predictGnbLOCAL))

# create a k neighbors classifier
kncLOCAL = KNeighborsClassifier(n_neighbors = 10)
# use the training metrics and results on the classifier
kncLOCAL.fit(trainMetricsLOCAL, trainResultLOCAL)
# predict a result using the test metrics
predictKncLOCAL = kncLOCAL.predict(testMetricsLOCAL)
# verify the accuracy between the testing results and the predicted values
print("Accuracy KNC: ", metrics.accuracy_score(testResultLOCAL, predictKncLOCAL))

print()

# ----- TRACE-OSB ----- #
# get 80% of the dataset using a seed to randomize the items for training
trainSetOSB = dataframeOSB.sample(frac = 0.8, random_state = 200)
# drop the selected items for the training and keep the remaining items for testing
testSetOSB = dataframeOSB.drop(trainSetOSB.index)
# get the testing result column
testResultOSB = testSetOSB.pop('success')
# keep the remaining columns for testing
testMetricsOSB = testSetOSB
# remove the result column to train what is to be expected
trainResultOSB = trainSetOSB.pop('success')
# keep the remaining columns for the training meters
trainMetricsOSB = trainSetOSB

# change the categorical columns of the training and testing meter sets to numerical
categoricalToNumerical(trainMetricsOSB, columnsGeneral)
categoricalToNumerical(testMetricsOSB, columnsGeneral)

print("trace_osb")

# create a decision tree classifier
treeOSB = tree.DecisionTreeClassifier()
# use the training metrics and results on the tree
treeOSB = treeOSB.fit(trainMetricsOSB, trainResultOSB)
# predict a result using the test metrics
predictTreeOSB = treeOSB.predict(testMetricsOSB)
# verify the accuracy between the testing results and the predicted values
print("Accuracy DTC: ", metrics.accuracy_score(testResultOSB, predictTreeOSB))

# create a support vector machine
svmOSB = svm.SVC(C = 1)
# use the training metrics and results on the machine
svmOSB.fit(trainMetricsOSB, trainResultOSB)
# predict a result using the test metrics
predictSvmOSB = svmOSB.predict(testMetricsOSB)
# verify the accuracy between the testing results and the predicted values
print("Accuracy SVM: ", metrics.accuracy_score(testResultOSB, predictSvmOSB))

# create a gaussian naive bayes
gnbOSB = GaussianNB()
# use the training metrics and results to predict a result
predictGnbOSB = gnbOSB.fit(trainMetricsOSB, trainResultOSB).predict(testMetricsOSB)
# verify the accuracy between the testing results and the predicted values
print("Accuracy GNB: ", metrics.accuracy_score(testResultOSB, predictGnbOSB))

# create a k neighbors classifier
kncOSB = KNeighborsClassifier(n_neighbors = 10)
# use the training metrics and results on the classifier
kncOSB.fit(trainMetricsOSB, trainResultOSB)
# predict a result using the test metrics
predictKncOSB = kncOSB.predict(testMetricsOSB)
# verify the accuracy between the testing results and the predicted values
print("Accuracy KNC: ", metrics.accuracy_score(testResultOSB, predictKncOSB))

print()

# ----- TRACE-RP ----- #
# get 80% of the dataset using a seed to randomize the items for training
trainSetRP = dataframeRP.sample(frac = 0.8, random_state = 200)
# drop the selected items for the training and keep the remaining items for testing
testSetRP = dataframeRP.drop(trainSetRP.index)
# get the testing result column
testResultRP = testSetRP.pop('success')
# keep the remaining columns for testing
testMetricsRP = testSetRP
# remove the result column to train what is to be expected
trainResultRP = trainSetRP.pop('success')
# keep the remaining columns for the training meters
trainMetricsRP = trainSetRP

# change the categorical columns of the training and testing meter sets to numerical
categoricalToNumerical(trainMetricsRP, columnsGeneral)
categoricalToNumerical(testMetricsRP, columnsGeneral)

print("trace_remote_process")

# create a decision tree classifier
treeRP = tree.DecisionTreeClassifier()
# use the training metrics and results on the tree
treeRP = treeRP.fit(trainMetricsRP, trainResultRP)
# predict a result using the test metrics
predictTreeRP = treeRP.predict(testMetricsRP)
# verify the accuracy between the testing results and the predicted values
print("Accuracy DTC: ", metrics.accuracy_score(testResultRP, predictTreeRP))

# create a gaussian naive bayes
gnbRP = GaussianNB()
# use the training metrics and results to predict a result
predictGnbRP = gnbRP.fit(trainMetricsRP, trainResultRP).predict(testMetricsRP)
# verify the accuracy between the testing results and the predicted values
print("Accuracy GNB: ", metrics.accuracy_score(testResultRP, predictGnbRP))

# create a k neighbors classifier
kncRP = KNeighborsClassifier(n_neighbors = 10)
# use the training metrics and results on the classifier
kncRP.fit(trainMetricsRP, trainResultRP)
# predict a result using the test metrics
predictKncRP = kncRP.predict(testMetricsRP)
# verify the accuracy between the testing results and the predicted values
print("Accuracy KNC: ", metrics.accuracy_score(testResultRP, predictKncRP))

print()