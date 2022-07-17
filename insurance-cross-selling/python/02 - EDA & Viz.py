# Databricks notebook source
# MAGIC %md
# MAGIC ## Explore, Analyse, Visualise
# MAGIC 
# MAGIC Digging into the data briefly, demonstrating some of the visualisation options available within a Databricks notebook.

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

# MAGIC %sql
# MAGIC SHOW DATABASES

# COMMAND ----------

# MAGIC %md
# MAGIC ### SQL
# MAGIC We can begin analysis using SQL (using `%sql`) because we have already registered a database, this can be viewed via the 'data' menu on the left-hand side

# COMMAND ----------

# MAGIC %sql
# MAGIC USE zdav_claims_cross_sell;
# MAGIC SHOW TABLES;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset
# MAGIC 
# MAGIC We have highly anonymised & modified data for ~40 thousands customers of this specific insurer seeking to enter car insurance business
# MAGIC 
# MAGIC | **Variable**           | **Definition**                                                                                                                  |
# MAGIC |--------------------|-----------------------------------------------------------------------------------------------------------------------------|
# MAGIC | id                 | Unique ID for the customer                                                                                                  |
# MAGIC | Gender               | Gender of the customer                                                                                                      |
# MAGIC | Age                | Age of the customer                                                                                                         |
# MAGIC | Driving_License    | **0**: Customer does not have DL<br>**1**: Customer already has DL                                                                  |
# MAGIC | Region_Code        | Unique code for the region of the customer                                                                                  |
# MAGIC | Previously_Insured | **1**: Customer already has Vehicle Insurance<br>**0**: Customer doesn't have Vehicle Insurance                                     |
# MAGIC | Vehicle_Age        | Age of the Vehicle                                                                                                          |
# MAGIC | Vehicle_Damage     | **1**: Customer got his/her vehicle damaged in the past<br>**0**: Customer didn't get his/her vehicle damaged in the past.          |
# MAGIC | Annual_Premium     | The amount customer needs to pay as premium in the year                                                                     |
# MAGIC | PolicySalesChannel | Anonymized Code for the channel of outreaching to the customer<br>ie. Different Agents, Over Mail, Over Phone, In Person, etc. |
# MAGIC | Vintage            | Number of Days, Customer has been associated with the company                                                               |
# MAGIC | Claims_Amount*      | Amount of money customer has claimed in the last year                                                                       |
# MAGIC | Claims_Num*        | Number of claims customer has submitted throughout their lifetime                                                           |
# MAGIC | **Response**\**          | **1**: Customer is interested<br>**0**: Customer is not interested                                                                  |
# MAGIC 
# MAGIC **\*** Features are created in notebook 01 for this walkthrough.  
# MAGIC **\*\*** `Response` is our target variable, all other columns are feature.
# MAGIC 
# MAGIC **Source**: [Health Insurance Cross Sell Prediction (Kaggle Dataset)](https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM zdav_claims_cross_sell.car_and_health_insurance_cross_sell
# MAGIC LIMIT 100;

# COMMAND ----------

# MAGIC %md
# MAGIC #### Is the annual premium influenced by age of the customer?

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT annual_premium, age
# MAGIC FROM zdav_claims_cross_sell.car_and_health_insurance_cross_sell
# MAGIC WHERE annual_premium > 500

# COMMAND ----------

# MAGIC %md
# MAGIC #### Identify the cross-sell opportunity for each region

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE VIEW cross_sell_region_agg AS
# MAGIC SELECT
# MAGIC   region_code,
# MAGIC   count(*) AS region_prospects,
# MAGIC   sum(annual_premium) AS total_annual_premium,
# MAGIC   sum(claims_amount) AS total_claims_amount,
# MAGIC   sum(claims_num) AS total_claims_count
# MAGIC FROM zdav_claims_cross_sell.car_and_health_insurance_cross_sell
# MAGIC GROUP BY region_code
# MAGIC ORDER BY region_code

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   region_code,
# MAGIC   region_prospects
# MAGIC FROM cross_sell_region_agg
# MAGIC ORDER BY region_prospects

# COMMAND ----------

# MAGIC %md
# MAGIC ### Switching gear, lets start using Python...
# MAGIC 
# MAGIC Choose your favourite data wrangling and visualisation libraries.  
# MAGIC If they are unavailable then install them to the attached cluster using these [instructions](https://docs.databricks.com/libraries/cluster-libraries.html#install-a-library-on-a-cluster).

# COMMAND ----------

# MAGIC %md
# MAGIC #### Collect basic statistics for each numeric feature

# COMMAND ----------

df = spark.table("zdav_claims_cross_sell.car_and_health_insurance_cross_sell")
cols = df.columns
numeric_cols = [c for c in cols if c not in ["id", "Gender", "Vehicle_Age", "Vehicle_Damage", "Region_Code", "Driving_License", "Previously_Insured", "Policy_Sales_Channel", "Response"]]
df.select(numeric_cols).describe().toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Calculate the percentage split for categorical features

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

pdf = df.toPandas()

prc_cols = ['Vehicle_Age', 'Gender', 'Vehicle_Damage', 'Driving_License', 'Previously_Insured']
sp_idx = 0
for i, col in enumerate(prc_cols):
  plt.subplot(1, 5, (i + 1))
  v_counts = pdf[col].value_counts()
  ax = v_counts.plot.pie(autopct = '%1.1f%%', figsize = (30,17), title = col + " %", textprops={'fontsize': 14})
  ax.yaxis.set_label_text('')

# COMMAND ----------

import plotly.express as px

pdf = spark.table("zdav_claims_cross_sell.cross_sell_region_agg").toPandas()

fig = px.scatter(pdf, x="total_claims_count", y="total_annual_premium", size="region_prospects",
                 hover_name="region_code", text="region_code", log_x=True, size_max=60)
fig.update_traces(textposition='bottom center')
fig.update_layout(title_text='Region-wise Claims and Annual Premium')
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cleanup

# COMMAND ----------

# MAGIC %sql
# MAGIC /* Remove view created at start of notebook */
# MAGIC DROP VIEW IF EXISTS cross_sell_region_agg

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Next Steps
# MAGIC [Previous Notebook - Data Preparation]($./01 - Insurance Data Preparation)  
# MAGIC [Next Notebook - Training A Model]($./03 - Predicting Interest In Cross Sell - mlflow)

# COMMAND ----------

