// Databricks notebook source
// importing relevant libs
// From here - Model preparation
import java.time._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
import org.apache.spark.sql.Column
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.joda.time.DateTime
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.evaluation.Evaluator

// COMMAND ----------

// load data (features only)
val boston_housing_dataset = spark.read.load("/mnt/S3/prod-parquet/product/daniel/meetup-data/")

// COMMAND ----------

// MAGIC %md
// MAGIC **Dataset Description**
// MAGIC 
// MAGIC **The Boston data frame has 506 rows and 14 columns**.
// MAGIC 
// MAGIC This dataset contains the following columns:
// MAGIC 
// MAGIC * **crim** - per capita crime rate by town.
// MAGIC * **zn** - proportion of residential land zoned for lots over 25,000 sq.ft.
// MAGIC * **indus** - proportion of non-retail business acres per town.
// MAGIC * **chas** - Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
// MAGIC * **nox** - nitrogen oxides concentration (parts per 10 million).
// MAGIC * **rm** - average number of rooms per dwelling.
// MAGIC * **age** - proportion of owner-occupied units built prior to 1940.
// MAGIC * **dis** - weighted mean of distances to five Boston employment centres.
// MAGIC * **rad** - index of accessibility to radial highways.
// MAGIC * **tax** - full-value property-tax rate per $10,000.
// MAGIC * **ptratio** - pupil-teacher ratio by town.
// MAGIC * **black** - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
// MAGIC * **lstat** - lower status of the population (percent).
// MAGIC * **price** - Price of house to predict (percentage)

// COMMAND ----------

// DBTITLE 1,Basic Statistics on the Dataframe
display(boston_housing_dataset.describe())

// COMMAND ----------

// DBTITLE 1,Train Test Split
val Array(trainingData, testData) = boston_housing_dataset.withColumnRenamed("PRICE", "label")
                                                          .randomSplit(Array(0.8, 0.2))

// COMMAND ----------

// DBTITLE 1,Get an Array of all column names (for vectorAssembler)
val relevantModelColumns = trainingData.drop("label").columns

// COMMAND ----------

// DBTITLE 1,Vector representation of relevant columns. Taking all of the relevant columns and getting a vector column called "features".
val assembler = new VectorAssembler()
  .setInputCols(relevantModelColumns)
  .setOutputCol("features")
  .setHandleInvalid("keep")

// COMMAND ----------

// DBTITLE 1,XGBoost Regressor (with objective reg:linear) instantiation, as it's a regression task
val xgboostRegressor = new XGBoostRegressor(Map[String, Any](
  "num_round" -> 80,
  "num_workers" -> 5,  //Distributed training
  "objective" -> "reg:linear",
  "eta" -> 0.1, // learning rate
  "gamma" -> 0.5,
  "max_depth" -> 6, 
  "early_stopping_rounds" -> 9,
  "seed" -> 1234, // for "holding" random results
  "lambda" -> 0.4,
  "alpha" -> 0.3,
  "colsample_bytree" -> 0.6,
  "subsample" -> 0.3
  ))

// COMMAND ----------

// DBTITLE 1,Create Pipeline of phases - 1. VectorAssembler -> 2. XGBoostRegressor
val pipeline = new Pipeline()
      .setStages(Array(assembler, 
                       xgboostRegressor))

// COMMAND ----------

// DBTITLE 1,Fit the training data on the pipeline
val trainedModel = pipeline.fit(trainingData)

// COMMAND ----------

// DBTITLE 1,Transform the Test Data (same as ".predict")
//transform on test data
val updatedAfterModel = trainedModel.transform(testData)

// create a tempView (ability to query with Spark-SQL API)
updatedAfterModel.createOrReplaceTempView("updatedAfterModel")

// COMMAND ----------

// DBTITLE 1,RegressionEvaluator - instantiating this one in order to have the ability to estimate our scores of the test data
val evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("mae") // there is support for MAE / RMSE / R2

val eval = evaluator.evaluate(updatedAfterModel)

// COMMAND ----------

// DBTITLE 1,SQL-Spark API on a DataFrame
// MAGIC %sql
// MAGIC SELECT label, prediction, ROUND(ABS(label - prediction), 4) absolute_diff FROM updatedAfterModel
// MAGIC ORDER BY absolute_diff ASC

// COMMAND ----------

// DBTITLE 1,SQL-Spark API on a DataFrame
// MAGIC %sql
// MAGIC SELECT * FROM updatedAfterModel