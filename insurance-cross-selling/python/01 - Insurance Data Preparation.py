# Databricks notebook source
# MAGIC %md
# MAGIC ## Data Preparation

# COMMAND ----------

# MAGIC %md ##### Quick tips before we get started..
# MAGIC * Attach the notebook to your cluster
# MAGIC * To run a cell, click Ctrl + Enter
# MAGIC * Press Escape, and then **h** to see keyboard shortcuts
# MAGIC * Click on the cell dashboard icon to add a chart to the Dashboard

# COMMAND ----------

# MAGIC %md
# MAGIC #### Explore Data Using Multiple Languages
# MAGIC Databricks notebooks support [multiple languages](https://docs.databricks.com/notebooks/notebooks-use.html#mix-languages).  
# MAGIC Each notebook has a default language.  
# MAGIC Use built-in 'magic' commands to switch over to another language or command.  
# MAGIC 
# MAGIC | Magic command |                  Purpose                       |
# MAGIC |---------------|------------------------------------------------|
# MAGIC | %md           |       Document using Markdown                  |
# MAGIC | %sql          |       Run a SQL command                        |
# MAGIC | %python       |       Run python code                          |
# MAGIC | %r            |       Run R code                               |
# MAGIC | %scala        |       Run Scala code                           |
# MAGIC | %fs           |       Shortcut for [dbutils.fs()](https://docs.databricks.com/dev-tools/databricks-utils.html#file-system-utilities)                     |
# MAGIC | %sh           |       Run a Bash shell command                 |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading Libraries
# MAGIC Databricks Notebooks supports Python out of the box and the [Databricks runtime](https://docs.databricks.com/runtime/index.html) already has many of the Python libaries you may use day to day.
# MAGIC 
# MAGIC For those that aren't its [easy to install them on the cluster](https://docs.databricks.com/libraries/cluster-libraries.html).

# COMMAND ----------

import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading Data
# MAGIC Databricks notebooks use the standard python you are familar with and therefore you can choose how to read in your data, here we will use `pd.read_csv` from `pandas`.  

# COMMAND ----------

# read in baseline data and have a quick look
# source: https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction?select=train.csv
raw_data = pd.read_csv("/dbfs/FileStore/zachary.davies@databricks.com/train.csv")

# COMMAND ----------

raw_data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Adjustments
# MAGIC 
# MAGIC Adding some additional features, specifically regarding the annual claim amount and annual premiums.  
# MAGIC **NOTE**: These features do not come with the original data and are generated to help the demo skip some complexity

# COMMAND ----------

def generate_annual_claim_amount(df, max_amount=120000):
    dist = np.random.lognormal(mean=0.0, sigma=1.75, size=df.shape[0])
    dist = dist * (max_amount / dist.max())
    dist[dist < 10] = 0.0
    dist = dist + (
      dist * (np.random.uniform(low=-0.2, high=0.2, size=dist.shape[0]))
    )
    
    response_positive_mask = df['Response'] == 1 
    dist[response_positive_mask] = dist[response_positive_mask] * np.random.uniform(low=1.8, high=3.0, size=dist[response_positive_mask].shape[0])
    
    dist = dist.astype(np.int32) 
    df['Claims_Amount'] =  dist
    return df
    
  
def generate_number_of_claims(df, avg_claim=40):
    nonzero_claims_mask = df['Claims_Amount'] > 0.0
    nonzero_arr = df['Claims_Amount'][nonzero_claims_mask].to_numpy()
    
    num_claims = (nonzero_arr / avg_claim) * np.random.uniform(low=1/3, high=1*3, size=nonzero_arr.shape[0])
    num_claims = num_claims - num_claims.min()
    
    num_claims[num_claims < 1.0] = 1.0
    num_claims = num_claims.astype(np.int32)
    
    df['Claims_Num'] = 0
    df.loc[nonzero_claims_mask, 'Claims_Num'] = num_claims
        
    return df

  
def adjust_annual_premium(df, avg_premium=40*52):
    m = df['Annual_Premium'].mean()
    factor = m / avg_premium
    
    df['Annual_Premium'] = df['Annual_Premium'] / factor
    return df

# COMMAND ----------

# applying adjustments
adj_data = generate_annual_claim_amount(raw_data)
adj_data = generate_number_of_claims(adj_data)
adj_data = adjust_annual_premium(adj_data)

# COMMAND ----------

# review data adjustments
adj_data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Storing Data
# MAGIC 
# MAGIC Databricks supports registering DataFrames as tables into databases.  
# MAGIC First a database must be created, here we can switch to a `%sql` chunk to do so.
# MAGIC 
# MAGIC Below we create the database and then show how to write our DataFrame.

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP DATABASE IF EXISTS zdav_claims_cross_sell CASCADE;
# MAGIC CREATE DATABASE IF NOT EXISTS zdav_claims_cross_sell;

# COMMAND ----------

spark\
  .createDataFrame(adj_data)\
  .repartition(1)\
  .write.format('delta')\
  .mode('overwrite')\
  .saveAsTable('zdav_claims_cross_sell.car_and_health_insurance_cross_sell')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Next Steps
# MAGIC 
# MAGIC [Previous Notebook - Introduction]($./00 - Introduction)  
# MAGIC [Next Notebook - Data Exploration]($./02 - EDA & Viz)