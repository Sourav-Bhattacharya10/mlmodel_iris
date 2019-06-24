# ML Model - Iris Classification Training
This code contains Python code to train a machine learning model Iris Classification. This code is scheduled to run once in every week using Crontab files in linux system.


## Quickstart
* Activate the virtual environment: **source app_env/bin/activate**
* Change to mlmodel_iris directory: **cd mlmodel_iris**
* To run the application in linux system: **python3 trainingscheduledjob/app.py**
* Deactivate the virtual environment: **deactivate**

OR

* Create a shell file 'modeltrainingscript.sh' with following content:

```shell
#!/bin/bash

################################
# Training Model using crontab
################################

cd /home/deviac/home/deployedmodels/irismodel/mlmodel_iris
../app_env/bin/python3 ./trainingscheduledjob/app.py
```

* Make the shell file executable with the following command:

```shell
chmod u+x modeltrainingscript.sh
```

* Run the shell file with following command:

```shell
./modeltrainingscript.sh
```

Now the project directory should look like this:

## Project Directory Structure

```shell
. (current directory - iris)
|   --- modeltrainingscript.sh
|   +-- app_env
|   +-- mlmodel_iris
|   |   +-- framework_models
|   |   +-- ml_pipeline
|   |   +-- predictionapi
|   |   +-- savedmodelfolder
|   |   +-- trainingscheduledjob
|   |   --- config.json
|   |   --- Iris.csv
|   |   --- README.md
|   |   --- requirements.txt
|   |   --- uwsgi_config.ini
```

## Deployment
For training the model, a scheduled job is created using crontab.

```shell
crontab -e
1 # Select Nano editor
```

As the editor opens the crontab file, write the following:

```shell
# Iris Classification Training Scheduled Job
0 0 * * 0 /home/deviac/home/deployedmodels/iris/modeltrainingscript.sh /home/deviac/home/logs/mlmodel_iris/python/cronlogfile.log 2>&1
```

Check the logfile.log and the cronlogfile.log in the following path : /home/deviac/home/logs/mlmodel_iris/python