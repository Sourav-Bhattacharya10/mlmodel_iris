#*************************************     LIBRARIES     ***************************************************#

# Default libraries
import json
import logging
from datetime import datetime

# Libraries that needs to be installed
import keras
import keras.backend.tensorflow_backend
import h5py
import tensorflow as tf


#***********************************************************************************************************#

with open("config.json") as g:
        gdata = json.load(g)

from ml_pipeline.logging import Logger

# Basic Logger
logger = Logger(logfilepath = gdata["logfilepath"])

# Save Keras Model
def saveEntireModel(model, modelfilepath = None):
    """
    Save Entire Keras model
    
    Params:\n
    `model` : Keras model object
    `modelfilepath` : If it is None, get it from the config.json Else pass the value
    """

    with open("config.json") as f:
        data = json.load(f)

    try:
        if modelfilepath == None:
            modelfilename = data["savedmodelfilename"]
            modelfilepath = str(data["modelfolder"]) + str(modelfilename)

        keras.models.save_model(model, modelfilepath)
        logger.writeToFile(str(datetime.today()) + ' : Saved keras model')

    except Exception as e:
        logger.writeToFile(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))

# Load Keras Model
def loadEntireModel(modelfilepath = None):
    """
    Load Entire Keras model
    
    Params:\n
    `modelfilepath` : If it is None, get it from the config.json Else pass the value

    Returns object model
    """

    with open("config.json") as f:
        data = json.load(f)

    model = None

    try:
        if modelfilepath == None:
            modelfilename = data["savedmodelfilename"]
            modelfilepath = str(data["modelfolder"]) + str(modelfilename)

        # # Unload Keras Model by reseting tf.Session
        # if keras.backend.tensorflow_backend._SESSION:
        #     tf.reset_default_graph() 
        #     keras.backend.tensorflow_backend._SESSION.close()
        #     keras.backend.tensorflow_backend._SESSION = None

        # logger.writeToFile(str(datetime.today()) + ' : Cleared tf.Session')

        model = keras.models.load_model(modelfilepath)
        logger.writeToFile(str(datetime.today()) + ' : Loaded keras model')
    
    except Exception as e:
        model = None
        logger.writeToFile(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))
    
    return model


def saveModelWeights(model, modelfilepath = None):
    """
    Save Keras model weights only
    
    Params:\n
    `model` : Keras model object
    `modelfilepath` : If it is None, get it from the config.json Else pass the value
    """

    with open("config.json") as f:
        data = json.load(f)

    try:
        if modelfilepath == None:
            modelfilename = data["savedmodelfilename"]
            modelfilepath = str(data["modelfolder"]) + str(modelfilename)

        file = h5py.File(modelfilepath, 'w')
        weight = model.get_weights()
        for i in range(len(weight)):
            file.create_dataset('weight' + str(i), data=weight[i])
        file.close()

        logger.writeToFile(str(datetime.today()) + ' : Saved keras model weights')
    
    except Exception as e:
        logger.writeToFile(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))


def loadModelWeights(model, modelfilepath = None):
    """
    Load Keras model weights only
    
    Params:\n
        `model` : Keras model object
    `modelfilepath` : If it is None, get it from the config.json Else pass the value

    Returns object model
    """

    with open("config.json") as f:
        data = json.load(f)

    try:
        if modelfilepath == None:
            modelfilename = data["savedmodelfilename"]
            modelfilepath = str(data["modelfolder"]) + str(modelfilename)

        file = h5py.File(modelfilepath, 'r')
        weight = []
        for i in range(len(file.keys())):
            weight.append(file['weight' + str(i)][:])
        model.set_weights(weight)

        logger.writeToFile(str(datetime.today()) + ' : Loaded keras model weights')
    
    except Exception as e:
        model = None
        logger.writeToFile(str(datetime.today()) + ' : Exception - ' + str(e.with_traceback))
    
    return model