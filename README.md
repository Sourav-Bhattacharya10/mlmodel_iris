# ML Model - Iris Classification
This contains five folders:

1. ml_pipeline : Contains helper functions to retrieve and process data faster.
2. framework_models : Conatins helper functions for saving and loading the models.
3. trainingscheduledjob : Contains python source code for training the machine learning model.
4. predictionapi : Contains python source code for consuming the trained machine learning model through REST API.
5. savedmodelfolder : Contains the files for the saved model.

## Home Directory Structure

```shell
.
+-- home
|   +-- deployedmodels
|   |   +-- iris
|   |   +-- project2
|   |   +-- project3
|   |   +-- project4
|   |   +-- project5
...
|   +-- logs
|   |   +-- mlmodel_iris
|   |   |   +-- python
|   |   +-- mlmodel_project2
|   |   |   +-- python
|   |   +-- mlmodel_project3
|   |   |   +-- python
|   |   +-- mlmodel_project4
|   |   |   +-- python
|   |   +-- mlmodel_project5
|   |   |   +-- python
...
```

Create this home directory structure manually.
Go to the **iris** directory and the execute the git clone command:

```shell
git clone https://github.com/Sourav-Bhattacharya10/mlmodel_iris.git
```

Now the project directory should look like this:

## Project Directory Structure

```shell
. (current directory - iris)
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


## Package Dependency
This project requires following packages (compulsory):
* xlrd
* numpy
* pandas
* pymssql
* pymongo
* scikit-learn
* keras
* h5py
* tensorflow
* Flask
* Flask-Cors
* uwsgi
* requests

**_NOTE_**: Do not remove any of the packages mentioned above as they are required for ml_pipeline and framework_models package. Add new packages at the end of the requirements.txt

**_NOTE_**: If using Ubuntu Server 18.04 LTS, run the following commands to create a virtual environment:

```shell
sudo apt-get install python3-venv
python3 -m venv app_env # to create a python virtual environment for uWSGI server.
```

Now the project directory should look like this:

## Project Directory Structure

```shell
. (current directory - iris)
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


**_NOTE_**: To run the training module or prediction module, the terminal should be inside the **mlmodel_iris** directory along with virtual environment activated. As the project configures itself based on the config.json file which is located inside the **mlmodel_iris** directory. If it cannot find the config.json, then it won't work.


Then

```shell
source app_env/bin/activate # to actiave the virtual environment.
cd mlmodel_iris
pip install --no-cache-dir -r requirements.txt
```

Once all the packages are installed, you are ready to work with the model.

Source code for training - mlmodel_iris/trainingscheduledjob/app.py
Source code for prediction - mlmodel_iris/predictionapi/app.py