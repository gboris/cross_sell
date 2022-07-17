# Databricks notebook source
# MAGIC %md
# MAGIC # Predicting Interest In Cross-sell

# COMMAND ----------

def get_username():
  username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
  return username

def get_notebook_path():
  notebook_path = str(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath()).replace('Some(', '').replace(')', '')
  return notebook_path

def get_folder_path():
  folder_path = '/'.join(notebook_path.split('/')[:-1])
  return folder_path

# COMMAND ----------

username = get_username()
notebook_path = get_notebook_path()
folder_path = get_folder_path()

print('username: "{}"'.format(username))
print('notebook_path: "{}"'.format(notebook_path))
print('folder_path: "{}"'.format(folder_path))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Creating An `mlflow` Experiment
# MAGIC 
# MAGIC Firstly, we need to create an mlflow `experiment`.  
# MAGIC Experiments contain multiple machine learning `runs` and are awesome for organisation & productivity.  
# MAGIC 
# MAGIC ![](https://databricks.com/wp-content/uploads/2019/10/model-registry-new.png)
# MAGIC 
# MAGIC Here is what we need to do:
# MAGIC - Run cell below 
# MAGIC - It will create a Machine Learning Experiment that will be located in `/cross-selling-insurance/experiments`
# MAGIC - Alternatively, click link the cell created.
# MAGIC - Observe empty Experiment. We will be pushing Machine Learning models to it soon! 

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from mlflow.utils.file_utils import TempDir

from sklearn.model_selection import train_test_split

import copy
import os
import json

# COMMAND ----------

experiment_location = os.path.join('/', folder_path, 'experiments', 'cross-selling-insurance')

try:
  experiment_id = mlflow.create_experiment(experiment_location)
except Exception as e:
  experiment_id = mlflow.get_experiment_by_name(experiment_location).experiment_id

# COMMAND ----------

displayHTML(f'<b>Experiment ID</b>: {experiment_id}<br><b>Experiment Location</b>: {experiment_location}<br><b>Experiment URL</b>: <a href="/#mlflow/experiments/{experiment_id}">/#mlflow/experiments/{experiment_id}</a></h4>')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Loading Dataset

# COMMAND ----------

# MAGIC %sql
# MAGIC USE zdav_claims_cross_sell;
# MAGIC SHOW TABLES;

# COMMAND ----------

data = spark.table("zdav_claims_cross_sell.car_and_health_insurance_cross_sell").toPandas()
display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC #### Correlation Matrix

# COMMAND ----------

plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Is the dataset balanced?
# MAGIC Looks like it is biased towards `0` as it has ~267k records vs `1` with ~37k records.

# COMMAND ----------

display(data.groupby('Response').count()[['id']])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Rebalance dataset
# MAGIC Lets use the larger cohort as a goal to resample up until for each group.  
# MAGIC This will leave the larger cohort (`0`) unchanged and adjust `1`.

# COMMAND ----------

# rebalance dataset, take under-represented outcome and sample to same size as other
df_balanced = data.groupby('Response').apply(pd.DataFrame.sample, n=sum(data.Response == 0), replace=True)
df_balanced.reset_index(drop=True, inplace=True)

# COMMAND ----------

display(df_balanced.groupby('Response').count()[['id']])

# COMMAND ----------

# MAGIC %md
# MAGIC #### How do features look after rebalancing?
# MAGIC Using notebook widgets we can make our exploration interactive.

# COMMAND ----------

correlated_features = ['Age', 'Previously_Insured', 'Policy_Sales_Channel', 'Claims_Amount', 'Claims_Num']
dbutils.widgets.dropdown('correlated_features', correlated_features[0], correlated_features)

# COMMAND ----------

fig = px.histogram(df_balanced.sample(10000), x=dbutils.widgets.get('correlated_features'), marginal='box', color='Response')
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initial Baseline Model
# MAGIC 
# MAGIC Will now create a baseline machine learning model to determine a performance baseline (good or bad).  
# MAGIC The task will be then to beat this benchmark with more advanced models/methods.  
# MAGIC 
# MAGIC Using `mlflow` we can ensure reproducilibty and track all our work.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Creating Training & Testing Splits
# MAGIC Splitting data into two portions:
# MAGIC 1. **Train:** 80% of data, used to train model
# MAGIC 2. **Test:** 20% of data, used for inference
# MAGIC 
# MAGIC Will be removing `id` from the data for modelling.

# COMMAND ----------

# random_state for reproducibility
# create partitions of 80%/20%
df_train, df_test = train_test_split(df_balanced, test_size=0.2, random_state=6789)

# COMMAND ----------

displayHTML(
  f'<b>all records</b>: {df_balanced.shape[0]}<br>' +\
  f'<b>training records</b>: {df_train.shape[0]}<br>' +\
  f'<b>testing records</b>: {df_test.shape[0]}'
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Training with `mlflow`
# MAGIC 
# MAGIC We will be using the `pyfunc` flavour of logging models to MLFlow.  
# MAGIC This is a special type of MLFlow 'flavour' that can handle anything!  
# MAGIC 
# MAGIC There are many flavours: `sklearn`, `tensorflow`, `keras`, among others. `pyfunc` is on a advanced side: if you can use it, you can use anything else.

# COMMAND ----------

import cloudpickle as pickle
from sklearn.metrics import f1_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# COMMAND ----------

class BaselineModel(mlflow.pyfunc.PythonModel):
  def __init__(self, random_forest_params={}, feature_columns=None):
    self._random_forest_params = random_forest_params
    self._feature_columns = feature_columns
  
  def fit(self, X, y):
    _X = copy.deepcopy(X)
    _X = _X[self._feature_columns]
    self._scaler = MinMaxScaler().fit(_X)
    _X = self._scaler.transform(_X)
    
    self._rf = RandomForestClassifier(**self._random_forest_params)
    self._rf.fit(_X, y)
    return self
      
  def predict(self, context, X):
    _y = self._predict(X)
    return _y 
    
  def _predict(self, X):
    _X = copy.deepcopy(X)
    _X = _X[self._feature_columns]
    _X = self._scaler.transform(_X)
    _y = self._rf.predict(_X)
    return _y
    
  def load_context(self, context):
    # this is where we define how to load "components" of the model
    with open(context.artifacts['feature_cols'], 'rb') as f:
      self._feature_columns = json.load(f)
      
    with open(context.artifacts['scaler'], 'rb') as f:
      self._scaler = pickle.load(f)
    
    with open(context.artifacts['random_forest'], 'rb') as f:
      self._rf = pickle.load(f)
      
  def log_to_mlflow(self):
    # this is the actual logging method: we define how to save "components" of the model
    with TempDir() as local_artifacts_dir:
      # dumping column names
      feature_cols_path = local_artifacts_dir.path('columns.json')
      with open(feature_cols_path, 'w') as m:
        json.dump(self._feature_columns, m)
      
      # dumping scaler
      scaler_path = local_artifacts_dir.path('scaler.pkl')
      with open(scaler_path, 'wb') as m:
        pickle.dump(self._scaler, m)
      
      # dumping model
      model_path = local_artifacts_dir.path('model.pkl')
      with open(model_path, 'wb') as m:
        pickle.dump(self._rf, m)
           
      # all of the model subcomponents will need to go here
      artifacts = {
        'feature_cols': feature_cols_path, 'scaler': scaler_path, 'random_forest': model_path
      }
      
      mlflow.pyfunc.log_model(
        artifact_path='model', python_model=self, artifacts=artifacts
      )

# COMMAND ----------

features = ['Age', 'Previously_Insured', 'Policy_Sales_Channel', 'Claims_Amount', 'Claims_Num']
label = 'Response'

X_train = df_train
y_train = df_train[label]

X_test = df_test
y_test = df_test[label]

with mlflow.start_run(experiment_id=experiment_id, run_name='baseline'):
  params = {'n_estimators': 200, 'max_depth': 7}
  print('Started run with params={}'.format(params))

  print('\ttraining on {} samples'.format(X_train.shape[0]))   
  model = BaselineModel(params, features).fit(X_train, y_train)
  print('\tdone training')
  
  print('\ttesting on {} samples'.format(X_test.shape[0]))
  y_pred_train = model._predict(X_train)
  y_pred_test = model._predict(X_test)
  print('\tdone inference')
  
  params = model._rf.get_params()
  mlflow.log_params(params)
  
  f1_train = f1_score(y_train, y_pred_train)
  print(f'\t-> f1_train={f1_train}')
  mlflow.log_metric('f1_train', f1_train)
  
  f1_test = f1_score(y_test, y_pred_test)
  print(f'\t-> f1_test={f1_test}')
  mlflow.log_metric('f1_test', f1_test)
  
  model.log_to_mlflow()

# COMMAND ----------

def gender_encoder(df, use_prefix=False):
  _mapping = {'Male': 0, 'Female': 1}
  
  new_col = 'Gender'
  if use_prefix:
    new_col = 'x_Gender' 
    
  df.loc[:, new_col] = df['Gender'].apply(lambda x: _mapping[x])
  return df
  
  
def vehicle_age_encoder(df, use_prefix=False):
  _mapping = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
  
  new_col = 'Vehicle_Age'
  if use_prefix:
    new_col = 'x_Vehicle_Age' 
    
  df.loc[:, new_col] = df['Vehicle_Age'].apply(lambda x: _mapping[x])
  return df
  
  
def vehicle_damage_encoder(df, use_prefix=False):
  _mapping = {'No': 0, 'Yes': 1}
  
  new_col = 'Vehicle_Damage'
  if use_prefix:
    new_col = 'x_Vehicle_Damage' 
    
  df.loc[:, new_col] = df['Vehicle_Damage'].apply(lambda x: _mapping[x])
  return df


funcs = [
  gender_encoder,
  vehicle_age_encoder,
  vehicle_damage_encoder,
]

for f in funcs:
  df_train = f(df_train, True)
  df_test = f(df_test, True)

# COMMAND ----------

display(df_train)

# COMMAND ----------

features = [
  'Age', 
  'Previously_Insured',
  'Policy_Sales_Channel',
  'Claims_Amount',
  'Claims_Num',
  'Driving_License',
  'Region_Code',
  'Annual_Premium',
  'Vintage',
  'x_Gender',
  'x_Vehicle_Age',
  'x_Vehicle_Damage'
]

label = 'Response'

X_train = df_train[features]
y_train = df_train[label]

X_test = df_test[features]
y_test = df_test[label]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Improving the Model
# MAGIC 
# MAGIC lets see if we can improve the model by trying  <img src="https://machinelearningapplied.com/wp-content/uploads/2019/10/xgboost_logo.png" alt="xgBoost" width="60", height="50"/>  and trying various parameters - we will use HyperOpt to do this for us in parallel.
# MAGIC 
# MAGIC ![my_test_image](https://www.jeremyjordan.me/content/images/2017/11/grid_search.gif)
# MAGIC ![my_test_image](https://www.jeremyjordan.me/content/images/2017/11/Bayesian_optimization.gif)

# COMMAND ----------

import copy
import cloudpickle as pickle
import xgboost as xgb
  
class XGBoostModel(mlflow.pyfunc.PythonModel):
  def __init__(self, xgb_params={}, feature_columns=None):
    self._xgb_params = xgb_params
    self._feature_columns = feature_columns
    
    self._transformations = [
      gender_encoder,
      vehicle_age_encoder,
      vehicle_damage_encoder
    ]
  
  def fit(self, X, y):
    _X = copy.deepcopy(X)
    for f in self._transformations:
      _X = f(_X)
      
    _X = _X[self._feature_columns]
    
    self._model = xgb.XGBClassifier(**self._xgb_params).fit(X=_X, y=y)
    return self
      
  def predict(self, context, X):
    _y = self._predict(X)
    return _y 
    
  def _predict(self, X):
    _X = copy.deepcopy(X)
    for f in self._transformations:
      _X = f(_X)
      
    _X = _X[self._feature_columns]
    
    _y = self._model.predict(_X)
    return _y
    
  def load_context(self, context):
    with open(context.artifacts['feature_cols'], 'rb') as f:
      self._feature_columns = json.load(f)
      
    with open(context.artifacts['transformations'], 'rb') as f:
      self._transformations = pickle.load(f)
    
    with open(context.artifacts['xgb'], 'rb') as f:
      self._model = pickle.load(f)
      
  def log_to_mlflow(self):
    with TempDir() as local_artifacts_dir:
      # dumping column names
      feature_cols_path = local_artifacts_dir.path('columns.json')
      with open(feature_cols_path, 'w') as m:
        json.dump(self._feature_columns, m)
      
      # dumping scaler
      transformations_path = local_artifacts_dir.path('transformations.pkl')
      with open(transformations_path, 'wb') as m:
        pickle.dump(self._transformations, m)
      
      # dumping model
      model_path = local_artifacts_dir.path('model.pkl')
      with open(model_path, 'wb') as m:
        pickle.dump(self._model, m)
           
      # all of the model subcomponents will need to go here
      artifacts = {
        'feature_cols': feature_cols_path, 'transformations': transformations_path, 'xgb': model_path
      }
      
      mlflow.pyfunc.log_model(
        artifact_path='model', python_model=self, artifacts=artifacts
      )

# COMMAND ----------

features = [
  'Age', 
  'Previously_Insured',
  'Policy_Sales_Channel',
  'Claims_Amount',
  'Claims_Num',
  'Driving_License',
  'Region_Code',
  'Annual_Premium',
  'Vintage',
  'Gender',
  'Vehicle_Age',
  'Vehicle_Damage'
]

X_train = df_train
y_train = df_train[label]

X_test = df_test
y_test = df_test[label]

# COMMAND ----------

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

fspace = {
    'max_depth': scope.int(hp.quniform('max_depth', 2, 11, q=1)),
    'reg_alpha': hp.uniform('reg_alpha', 0.001, 0.1),
    'reg_lambda': hp.uniform('reg_lambda', 0.001, 0.1), 
    'n_estimators': scope.int(hp.quniform('n_estimators', 50, 250, q=1))
}

def f(params):
  with mlflow.start_run(experiment_id=experiment_id, run_name='HyperOpt-XGBoost'):
    model = XGBoostModel(xgb_params=params, feature_columns=features).fit(X_train, y_train)
    model.log_to_mlflow()
    
    y_pred_test = model._predict(X_test)
    y_pred_train = model._predict(X_train)
    
    f1_test = f1_score(y_test, y_pred_test)
    f1_train = f1_score(y_train, y_pred_train)
    mlflow.log_metrics({'f1_train': f1_train, 'f1_test': f1_test})
    mlflow.log_params(model._model.get_params())
    
    return {'loss': -f1_test, 'status': STATUS_OK}

trials = Trials()
best = fmin(fn=f, space=fspace, algo=tpe.suggest, max_evals=10, trials=trials)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Fetch Experiment Runs

# COMMAND ----------

all_runs = mlflow.search_runs(experiment_ids=[experiment_id])
display(all_runs)

# COMMAND ----------

best_runs= mlflow.search_runs(experiment_ids=[experiment_id], filter_string='metrics.f1_test > 0.25', order_by=['metrics.f1_test desc'])
print(f'Found {best_runs.shape[0]} OK runs')

best_run = best_runs.iloc[0]
print(best_run)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Best Model & Predict

# COMMAND ----------

model = mlflow.pyfunc.load_model(best_run['artifact_uri'] + '/model')

# COMMAND ----------

model.predict(data)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Prediction with UDFs

# COMMAND ----------

# using pyfunc.spark_udf we can return a UDF
pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_uri=best_run['artifact_uri'] + '/model')

# COMMAND ----------

adj_data2 = pd.concat([X_train, X_test])

spark\
  .createDataFrame(adj_data2)\
  .repartition(4)\
  .write.format('delta')\
  .mode('overwrite')\
  .saveAsTable('zdav_claims_cross_sell.inference_data')

# COMMAND ----------

from pyspark.sql.functions import struct
predicted_df = spark.table("inference_data").withColumn("prediction", pyfunc_udf(struct(*features)))

# COMMAND ----------

predicted_df.toPandas()

# COMMAND ----------

predicted_df\
  .write.format('delta')\
  .mode('overwrite')\
  .saveAsTable('zdav_claims_cross_sell.data_with_predictions')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Registry

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
from pprint import pprint


client = MlflowClient()
for mv in client.search_model_versions("name='zdav-cross-sell'"):
    pprint(dict(mv), indent=4)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cleanup

# COMMAND ----------

# remove widgets made during notebook
dbutils.widgets.removeAll()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Finished!
# MAGIC [Previous Notebook - Data Exploration]($./02 - EDA & Viz)