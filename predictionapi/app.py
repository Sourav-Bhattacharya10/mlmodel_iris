#*************************************     LIBRARIES     ***************************************************#

# Default libraries
import sys
import json
import logging
from datetime import datetime

# Libraries that needs to be installed
import numpy as np
import pandas as pd
import keras
import keras.backend as Kb
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS

# User defined libraries
with open("config.json") as g:
    gdata = json.load(g)

sys.path.append(gdata["mlpipelinepath"])

from ml_pipeline.data_processing import normalizeData
from framework_models.keras_model import loadModelWeights


#***********************************************************************************************************#


# Basic Logger
logging.basicConfig(filename = gdata["logfilepath"], level = logging.INFO)


# Enable Cross Origin Resource Sharing (CORS)
application = Flask(__name__)
CORS(application)


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

        logging.info(str(datetime.today()) + ' : TRAIN - Compiled model')

    except Exception as e:

        model = None
        logging.exception(str(datetime.today()) + ' : TRAIN - Exception - ' + str(e.with_traceback))
    
    return model


#***********************************************************************************************************#


#****************************************     METHODS     **************************************************#


# NOTHING


#***********************************************************************************************************#


#*****************************************     PREDICT     *************************************************#


# Default Prediction API
# POST /iris/predict
# Request Header Content-Type : text/plain
# BODY {"petal_lengths" : ["4.0", "7.0", "6.3"], "petal_widths" : ["2.8", "3.2", "2.7"], "sepal_lengths" : ["1.0", "4.7", "4.9"], "sepal_widths" : ["0.1", "1.4", "1.8"]}
@application.route("/iris/predict", methods=["POST"])
def predictDefault():

    logging.info(str(datetime.today()) + ' : PREDICT - Calling default prediction API')

    # Load the json file
    with open("config.json") as f:
        data = json.load(f)

    try:
        reqDict = json.loads(request.data)

        petal_lengths = reqDict['petal_lengths']
        petal_lengths = [float(x) for x in petal_lengths]

        petal_widths = reqDict['petal_widths']
        petal_widths = [float(x) for x in petal_widths]

        sepal_lengths = reqDict['sepal_lengths']
        sepal_lengths = [float(x) for x in sepal_lengths]

        sepal_widths = reqDict['sepal_widths']
        sepal_widths = [float(x) for x in sepal_widths]

        df1 = pd.DataFrame({'petal_length':petal_lengths, 'petal_width':petal_widths, 'sepal_length':sepal_lengths, 'sepal_width':sepal_widths})
        
        logging.info(str(datetime.today()) + ' : PREDICT - Got data')

        Xdata = df1.values

        logging.info(str(datetime.today()) + ' : PREDICT - Got Xdata')

        Xdata = normalizeData(Xdata, fit_transform = False)

        logging.info(str(datetime.today()) + ' : PREDICT - Normalized Xdata using MinMaxScaler with fit transform as False')

        with tf.Session(graph=tf.Graph()) as sess:
            Kb.set_session(sess)
            
            #Instantiate the model
            model = MyKerasModel(num_input = 4, num_h1 = 5, num_h2 = 6, num_output = 3)
            
            #Load the model
            model = loadModelWeights(model)
            #model = loadEntireModel()
            
            preds = model.predict(Xdata)

        logging.info(str(datetime.today()) + ' : PREDICT - Loaded the model and predicted the unknown data')

        predictedcategory = [np.argmax(x) for x in preds]

        logging.info(str(datetime.today()) + ' : PREDICT - Found the np.argmax of all rows')

        iris_names = ["Iris-virginica", "Iris-setosa", "Iris-versicolor"]
        predictions = [iris_names[x] for x in predictedcategory]

        logging.info(str(datetime.today()) + ' : PREDICT - Assigned categories to the predictions')

        # Read the RMSE value of the test set and append it with the prediction
        metricfile = str(data["modelfolder"]) + str(data["metricfilepath"])
        with open(metricfile) as t:
            tmpdata = json.load(t)

        logging.info(str(datetime.today()) + ' : PREDICT - Read RMSE value')

        return jsonify(testaccuracy = tmpdata["testaccuracy"], data = predictions)

    except Exception as e:
        
        logging.exception(str(datetime.today()) + ' : PREDICT - Exception - ' + str(e.with_traceback))
        return jsonify(error=str(e))


if __name__ == "__main__":
    application.run(debug=False)