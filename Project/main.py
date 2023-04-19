import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import pandas as pd
from keras.applications import VGG16 #for features extraction using cnn
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy import ndimage as nd
from skimage.filters import sobel
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import svm
import random

def loadDataFromKeras():
    image_data = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = image_data.load_data()
    # reshape dataset to have a single channel
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    # normalize to range 0-1
    train_norm = train_images / 255.0
    test_norm = test_images / 255.0
    return train_images, train_labels, test_images, test_labels

def loadDataFromCSV():
    fashion_mnist = pd.read_csv("mnistTrain.csv")
    # Split the dataset into features (X) and target (y)
    train_images = fashion_mnist.iloc[:, 1:].values
    train_labels = fashion_mnist.iloc[:, 0].values
    fashion_mnisttest = pd.read_csv("mnistTest.csv")
    test_images = fashion_mnisttest.iloc[:, 1:].values
    test_labels = fashion_mnisttest.iloc[:, 0].values
    return train_images, train_labels, test_images, test_labels

def validation(trainFeatures,trainLabels):
    X_train, X_val, y_train, y_val = train_test_split(trainFeatures, trainLabels, test_size=(1 / 3), shuffle=True)
    return X_train, X_val, y_train, y_val
def featuresExtraction(x_train,x_test):
    # Flatten the 28x28 images into 1D arrays of 784 values
    x_train = x_train.reshape(len(x_train), 784)
    x_test = x_test.reshape(len(x_test), 784)
    # Normalize the pixel intensities to the range [0, 1]
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Use the flattened pixel intensities as features for the dataset
    features_train = x_train
    features_test = x_test
    return  features_train,features_test
def visualizeConfusionMatrix(cm,set):
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix For "+set)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.colorbar()
    plt.show()

def baseLine():
    #loading data
    train_images, train_labels, test_images, test_labels=loadDataFromCSV()
    #features extraction
    features_train, features_test=featuresExtraction(train_images,test_images)
    #validation set
    xTrainVal, xTestVal,yTrainVal, yTestVal=validation(features_train,train_labels)
    #best case of evaluating knn if for k=1 and distance="eucladian" from these sets: k=(1,3) and distances=("eucladian","manhattan")
    #so
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    knn.fit(xTrainVal, yTrainVal)
    y_predVal = knn.predict(xTestVal)
    accuracyVal = accuracy_score(yTestVal, y_predVal)
    cm = confusion_matrix(yTestVal, y_predVal)
    #baseLine on the real testing data
    knn.fit(features_train,train_labels)
    trainpred=knn.predict(features_train)
    accuracyTrain=accuracy_score(train_labels,trainpred)
    cm3 = confusion_matrix(train_labels,trainpred)
    print("Accuracy for validation set=", accuracyVal)
    # print("Confusion matrix for validation set:",cm)
    print("Accuracy for training data=",accuracyTrain)
    #print("Confustion matrix for training data: ",cm3)
    y_pred=knn.predict(features_test)
    accuracy=accuracy_score(test_labels,y_pred)
    cm2 = confusion_matrix(test_labels,y_pred)
    print("Accuracy for testing data=",accuracy)
    #print("Confusion matrix for testing data=", cm2)
    visualizeConfusionMatrix(cm,"Validation")
    visualizeConfusionMatrix(cm2,"Training")
    visualizeConfusionMatrix(cm3,"Testing")


def get_features(model, layer_name, example):
    intermediate_layer_model = tf.keras.models.Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
    features = intermediate_layer_model.predict(example[np.newaxis, ...])
    return features
def incorrectIndices(y_pred,y_test,x_test):
    incorrect_indices = np.where(y_pred != y_test)[0]
    # Select 20 random images from the incorrect predictions
    random_incorrect_indices = np.random.choice(incorrect_indices, size=20)
    # Plot the selected images along with their true labels and predicted labels
    for i, incorrect_index in enumerate(random_incorrect_indices):
        plt.subplot(4, 5, i + 1)
        plt.imshow(x_test[incorrect_index][..., 0], cmap='gray')
        plt.axis('off')
    plt.show()
def cnn():
    train_images, train_labels, test_images, test_labels=loadDataFromKeras()
    # validation set
    xTrainVal, xTestVal, yTrainVal, yTestVal = validation(train_images, train_labels)
    #features extraction is done inside the sequential
    '''model =tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128,activation="relu"),
        tf.keras.layers.Dense(10)# numbers within the dataset e.i. clothes types
    ])'''
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer="adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"])
    #first fitting for validation set
    model.fit(xTrainVal,yTrainVal,epochs=10)
    test_loss_val, accuracyVal = model.evaluate(xTestVal, yTestVal, verbose=0)
    model.fit(train_images,train_labels,epochs=10)
    train_loss, accuracyTrain = model.evaluate(train_images, train_labels, verbose=0)
    test_loss, accuracyTest = model.evaluate(test_images, test_labels, verbose=0)
    print("Accuracy for validation set=", accuracyVal)
    print("Loss =", test_loss_val)
    print("Accuracy for training data=", accuracyTrain)
    print("Loss =", train_loss)
    print("Accuracy for testing data=", accuracyTest)
    print("Loss =", test_loss)
    #====================================================================================
    # get 20 missed examples of testing data and visualize it
    pred_y=model.predict(test_images)
    incorrectIndices(pred_y,test_labels,test_images)
    #====================================================================================
    # ====================================================================================
    # Make predictions on the test set
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    # Get the indices of the incorrect predictions
    incorrect_indices = np.where(predicted_classes != test_labels)[0]
    # Select the first incorrect example
    example = test_images[incorrect_indices[0]]
    features = get_features(model, 'conv2d_1', example)
    mean_feature = np.mean([get_features(model, 'conv2d_1', test_images[index]) for index in incorrect_indices], axis=0)
    '''# Plot the mean feature
    plt.imshow(mean_feature[0, ..., 0], cmap='gray')
    plt.show()'''
    # Create an input layer
    input_layer = tf.keras.layers.Input(shape=(28, 28, 1))
    # Add the mean feature layer
    mean_feature_layer = tf.keras.layers.Conv2D(1, (3, 3), padding='same', activation='relu')(input_layer)
    conv_layer = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mean_feature_layer)
    flat_layer = tf.keras.layers.Flatten()(conv_layer)
    dense_layer = tf.keras.layers.Dense(128, activation='relu')(flat_layer)
    output_layer = tf.keras.layers.Dense(10, activation='softmax')(dense_layer)
    # Define the model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.fit(train_images,train_labels)
    new_test_loss, new_accuracyTest = model.evaluate(test_images, test_labels, verbose=0)
    print("Accuracy of testing data for improved model=",new_accuracyTest)
    print("Accuracy of testing data for improved loss=", new_test_loss)
    # ====================================================================================


def svmModel():
    train_images, train_labels, test_images, test_labels = loadDataFromKeras()
    # features extraction
    features_train, features_test = featuresExtraction(train_images, test_images)
    # validation set
    xTrainVal, xTestVal, yTrainVal, yTestVal = validation(features_train, train_labels)
    # Train the SVM classifier
    clf = svm.SVC(kernel='linear')
    clf.fit(xTrainVal, yTrainVal)
    # Evaluate the classifier on the validation set
    y_predVal = clf.predict(xTestVal)
    accuracyVal = accuracy_score(yTestVal, y_predVal)
    clf.fit(features_train, train_labels)
    y_predTrain = clf.predict(features_train)
    accuracyTrain = accuracy_score(train_labels, y_predTrain)
    y_predTest = clf.predict(features_test)
    accuracyTest = accuracy_score(test_labels, y_predTest)
    # Plot the confusion matrix
    cm1 = confusion_matrix(yTestVal, y_predVal)
    cm2 = confusion_matrix(train_labels, y_predTrain)
    cm3 = confusion_matrix(test_labels, y_predTest)
    visualizeConfusionMatrix(cm1, "Validation set")
    visualizeConfusionMatrix(cm2, "Training set")
    visualizeConfusionMatrix(cm3, "Testing set")
    print("Accuracy for Validation set=",accuracyVal)
    print("Accuracy for Training set=", accuracyTrain)
    print("Accuracy for Testing set=", accuracyTest)
    # ====================================================================================
    # get 20 missed examples of testing data and visualize it
    incorrectIndices(y_predTest, test_labels, test_images)
    # ====================================================================================
    # ====================================================================================
    #get incorrect data
    incorrect_idx = np.where(y_predTest != test_labels)[0]

    # get the feature vectors of the incorrectly predicted data
    incorrect_data = features_test[incorrect_idx]


    # update the model with new data
    features_train = np.concatenate((features_train, incorrect_data))
    train_labels = np.concatenate((train_labels, test_labels[incorrect_idx]))
    clf.fit(features_train, train_labels)

    # re-evaluate the accuracy of the model
    y_predTest = clf.predict(features_test)
    accuracy = accuracy_score(test_labels, y_predTest)
    print("Accuracy after updating the model:", accuracy)
    # ====================================================================================

#def improvedCNN():

#baseLine()
#cnn()
#svmModel()



