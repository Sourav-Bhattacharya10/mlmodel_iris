#*************************************     LIBRARIES     ***************************************************#

# Default libraries
import sys
import json
from datetime import datetime

# Libraries that needs to be installed
import numpy as np
import pandas as pd
import keras
import keras.backend as Kb
import tensorflow as tf

# User defined libraries
with open("config.json") as g:
    gdata = json.load(g)

sys.path.append(gdata["mlpipelinepath"])

from ml_pipeline.logging import Logger
from ml_pipeline.data_input import inputData
from ml_pipeline.data_processing import processData
from ml_pipeline.data_processing import dummifyData
from ml_pipeline.data_processing import getObjectTypeColumns
from ml_pipeline.data_processing import divideDataInXY
from ml_pipeline.data_processing import normalizeData
from ml_pipeline.data_processing import generateTrainAndTestXY
from ml_pipeline.data_processing import getUniqueValuesDictionary
from framework_models.keras_model import saveModelWeights


#***********************************************************************************************************#


# Basic Logger
logger = Logger(logfilepath = gdata["logfilepath"])


#****************************************     MODEL     ****************************************************#

def MyKerasModel(num_input = 1, num_h1 = 1, num_h2 = 1, num_output = 3):
    # Keep your model code here
    model = None
    
    try:
        init = keras.initializers.glorot_uniform(seed=1)
        simple_adam = keras.optimizers.Adam()
        
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(units=num_h1, input_dim=num_input, kernel_initializer=init, activation='relu'))
        model.add(keras.layers.Dense(units=num_h2))
        model.add(keras.layers.Dense(units=num_output, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])

        logger.writeToFile(str(datetime.today()) + ' : TRAIN - Compiled model')

    except Exception as e:

        model = None
        logger.writeToFile(str(datetime.today()) + ' : TRAIN - Exception - ' + str(e.with_traceback))
    
    return model


#***********************************************************************************************************#


#****************************************     METHODS     **************************************************#


# NOTHING


#***********************************************************************************************************#


#****************************************     TRAINING     *************************************************#

# Method for training the model based on seven days time interval
def training():

    np.random.seed(4)
    tf.set_random_seed(2)

    # Load the json file
    with open("config.json") as f:
        data = json.load(f)

    try:

        # Fetch data using inputData method of ml_pipeline
        df1 = inputData("csv")
        
        logger.writeToFile(str(datetime.today()) + ' : TRAIN - Got data')

        df2 = processData(df1, columnsNamesToWorkWith = ['petal_length','petal_width','sepal_length','sepal_width','label'], shuffleData = True, substituteEmptyFields = False)

        logger.writeToFile(str(datetime.today()) + ' : TRAIN - Filtered columns to work with and shuffled data')

        colslist = getObjectTypeColumns(df2)

        logger.writeToFile(str(datetime.today()) + ' : TRAIN - Got Object type columns for one hot encoding')

        uvDict = getUniqueValuesDictionary(df2, colslist)

        logger.writeToFile(str(datetime.today()) + ' : TRAIN - Got unique values from the object type columns')

        df3 = dummifyData(df2, uvDict)

        logger.writeToFile(str(datetime.today()) + ' : TRAIN - Dummified data')

        listcolYNames = ['label_' + x for x in uvDict['label']]

        logger.writeToFile(str(datetime.today()) + ' : TRAIN - Got one hot encoded column names for Y dataset')

        Xdata, Ydata = divideDataInXY(df3, listcolYNames)

        logger.writeToFile(str(datetime.today()) + ' : TRAIN - Divided data in X and Y')

        Xdata = normalizeData(Xdata, fit_transform = True)

        logger.writeToFile(str(datetime.today()) + ' : TRAIN - Normalized Xdata using MinMaxScaler with fit transform as True')

        trainX, trainY, testX, testY = generateTrainAndTestXY(Xdata, Ydata, splitfraction = 0.9)

        logger.writeToFile(str(datetime.today()) + ' : TRAIN - Split data in train and test sets')

        with tf.Session(graph=tf.Graph()) as sess:
            Kb.set_session(sess)
            
            #Instantiate the model
            model = MyKerasModel(num_input = 4, num_h1 = 5, num_h2 = 6, num_output = 3)

            #Train the model
            history = model.fit(trainX, trainY, epochs = 300, batch_size = 10, validation_split= 0.2, verbose = 0)
            
            #Evaluate the model
            eval1 = model.evaluate(trainX, trainY, verbose=0)
            trainScore = eval1[1] * 100

            eval2 = model.evaluate(testX, testY, verbose=0)
            testScore = eval2[1] * 100
            
            #Save the model weights
            saveModelWeights(model)

        logger.writeToFile(str(datetime.today()) + ' : TRAIN - Trained, evaluated and saved the model')

        logger.writeToFile(str(datetime.today()) + ' : TRAIN - TrainLoss is ' + str(trainScore) + ' and TestLoss is ' + str(testScore))

        # Write the Metric value
        metricfile = str(data["modelfolder"]) + str(data["metricfilepath"])
        with open(metricfile, 'w+') as outfile:
            json.dump({"trainaccuracy": str(trainScore), "testaccuracy" : str(testScore)}, outfile)

        logger.writeToFile(str(datetime.today()) + ' : TRAIN - Wrote Metric values')

    except Exception as e:
        
        logger.writeToFile(str(datetime.today()) + ' : TRAIN - Exception - ' + str(e.with_traceback))
        


#***********************************************************************************************************#


training()