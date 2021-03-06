// Databricks notebook source
// DBTITLE 1,Importing relevant libraries & functions
// ML imports
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.feature.VectorAssembler
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
import org.apache.spark.ml.Pipeline

// used for model evaluation
import org.apache.spark.ml.evaluation.RegressionEvaluator

// Data imports & manipulation
import org.apache.spark.sql.Column
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

// MLeap model saving
import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport._
import org.apache.spark.ml.bundle.SparkBundleContext
import ml.combust.mleap.runtime.MleapSupport._
import resource._

// COMMAND ----------

// DBTITLE 1,Data Loading (Note: It's saved on our S3, but available online, e.g.: https://www.kaggle.com/c/boston-housing)
val boston_housing_dataset = spark.read.load("/mnt/S3/prod-parquet/product/daniel/meetup-data/")

// COMMAND ----------

// MAGIC %md
// MAGIC ** Boston Housing Dataset Description **
// MAGIC 
// MAGIC ***The dataframe has 506 rows and 14 columns, in which below you can find a description about those:***
// MAGIC 
// MAGIC <br>
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

// DBTITLE 1,Add Dummy columns (both string and Int columns) to data
val boston_housing_dataset_dummy = boston_housing_dataset.withColumn("dummyString", concat(lit("dummy"),round(rand(seed=42)*10,0)))
                                                         .withColumn("dummyInt",round(rand(seed=10),2))

// COMMAND ----------

display(boston_housing_dataset_dummy)

// COMMAND ----------

// DBTITLE 1,Add missing data to the Dataframe and handle it in XGBoost
val boston_with_dummy_and_missing = boston_housing_dataset_dummy.withColumn("LSTAT_MISSING", when(col("LSTAT") <= 5, null).otherwise(col("LSTAT")))
                                                                .drop("LSTAT")
                                                                .withColumnRenamed("LSTAT_MISSING", "LSTAT")

// COMMAND ----------

display(boston_with_dummy_and_missing)

// COMMAND ----------

// DBTITLE 1,Handling missing values. make sure to put a dummy value on them (-99.0 on this case)
// Replace null values with a missing value -99.0 as the relevant cols that have missing columns are all double
val replacedValuesDf = boston_with_dummy_and_missing.na.fill(-99.0)
// val finalDf = descDf.describe()

// COMMAND ----------

display(replacedValuesDf)

// COMMAND ----------

// DBTITLE 1,Train Test Split
val Array(trainingData, testData) = replacedValuesDf.withColumnRenamed("PRICE", "label")
                                                    .randomSplit(Array(0.8, 0.2))

// COMMAND ----------

// DBTITLE 1,String Indexer on String Categorical Columns (dummyString)
val dummyIndexer = new StringIndexer()
  .setInputCol("dummyString")
  .setOutputCol("dummyString_indexed")

// COMMAND ----------

// DBTITLE 1,oneHotEncoding on categorical data
val dummyOneHotEncoded = new OneHotEncoderEstimator()
  .setInputCols(Array("dummyString_indexed"))
  .setOutputCols(Array("dummyString_oneHotEncoded"))

// COMMAND ----------

// DBTITLE 1,Get an Array of all relevant columns for training (this is for VectorAssembler)
// note - added manipulated column as well.
val relevantModelColumns = (trainingData.drop("label", "dummyString").columns) ++  Array("dummyString_oneHotEncoded")

// COMMAND ----------

// DBTITLE 1,Vector representation of relevant columns. Taking all of the relevant columns and getting a vector column called "features".
val assembler = new VectorAssembler()
  .setInputCols(relevantModelColumns)
  .setOutputCol("features")
  .setHandleInvalid("keep")

// COMMAND ----------

// DBTITLE 1,XGBoost Regressor (with objective reg:linear) instantiation, as it's a regression task. Note: missing flag = -99.0
val xgboostRegressor = new XGBoostRegressor(Map[String, Any](
  "num_round" -> 80,
  "missing" -> -99.0, //As we do have missing values in the dataset!
  "num_workers" -> 5,  //Distributed training
  "objective" -> "reg:linear",
  "eta" -> 0.1, // learning rate
  "max_depth" -> 6,
  "seed" -> 1234, // for "holding" random results
  "lambda" -> 0.4,
  "alpha" -> 0.3,
  "colsample_bytree" -> 0.6,
  "subsample" -> 0.3
  ))

// COMMAND ----------

// DBTITLE 1,Create Pipeline of phases - 1. StringIndexer, 2. OneHotEncoding, 3. VectorAssembler, 4. XGBoostRegressor
val pipeline = new Pipeline()
                      .setStages(Array(dummyIndexer,
                                       dummyOneHotEncoded,
                                       assembler,
                                       xgboostRegressor))

// COMMAND ----------

// DBTITLE 1,Fit the training data on the pipeline
val trainedModel = pipeline.fit(trainingData)

// COMMAND ----------

// DBTITLE 1,Model Saving using MLeap (https://mleap-docs.combust.ml/getting-started/spark.html)
// this transform is solely for model saving - not used for evaluation!
val updatedDfTrainedAfterModel = trainedModel.transform(trainingData)

// COMMAND ----------

// DBTITLE 1,Making an empty directory for the model
// MAGIC %sh
// MAGIC rm -rf /tmp/mleap_xgboost_meetup/
// MAGIC mkdir /tmp/mleap_xgboost_meetup/

// COMMAND ----------

// DBTITLE 1,Save Model (preparation)
implicit val context = SparkBundleContext().withDataset(updatedDfTrainedAfterModel)

// COMMAND ----------

// DBTITLE 1,Serialize Pipeline to Directory
(for(modelFile <- managed(BundleFile("file:/tmp/mleap_xgboost_meetup"))) yield {
  trainedModel.writeBundle.save(modelFile)(context)
}).tried.get

// COMMAND ----------

// DBTITLE 1,Save the file in S3
dbutils.fs.cp("file:/tmp/mleap_xgboost_meetup/", s"/mnt/S3/prod-parquet/product/daniel/xgboost-meetup/model", recurse=true)
// of course, for version management, we can add some timestamp in the format of /day=yyyy-MM-dd/hour=HH, just to make our life easier

// COMMAND ----------

// DBTITLE 1,(Optional) Load the model after saved it
val dirBundle = (for(bundle <- managed(BundleFile("file:/tmp/mleap_xgboost_meetup"))) yield {
  bundle.loadSparkBundle().get
}).opt.get

// COMMAND ----------

// DBTITLE 1,Get our ML pipeline out of it
val relevantPipeline = dirBundle.root

// COMMAND ----------

// DBTITLE 1,And make predictions (this can be expanded to REST API services, batch inference, etc.)
val predDf = relevantPipeline.transform(testData)

// COMMAND ----------

// DBTITLE 1,Getting back to the normal flow. Using the "trainedModel" in order to transform the Test Data (i.e. ".predict")
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
// MAGIC SELECT label, prediction, ROUND(ABS(label - prediction), 4) absolute_diff 
// MAGIC FROM updatedAfterModel
// MAGIC ORDER BY absolute_diff ASC

// COMMAND ----------

// DBTITLE 1,SQL-Spark API on a whole DataFrame (foucsing on features column) 
// MAGIC %sql
// MAGIC SELECT * FROM updatedAfterModel
// MAGIC -- features
// MAGIC -- 0: this is the first index that says it is: 1 - a Dense Representation (more non-zero than zero elements - done automatically by spark); 0 - Sparse Representation
// MAGIC -- 1: this is the size of the feature vector row
// MAGIC -- 2: The indices of the Sparse representation - meaning which features are non-zero in a Sparse representation (thus will always be empty in Dense repersentation)
// MAGIC -- 3: the actual values of the indices that were found in 2

// COMMAND ----------

// DBTITLE 1,Feature Importance #1
// Feature importance is not that easy in XGBoost4J (the Spark version)
// In order to get it, one should dive into the low-level APIs of the model, and get the relevant method out of it, then join the scores with the column names

// getting the low-level API of xgboostRegressionModel in order to get the features and feature scores
// 1. get the model out of it
val model = trainedModel.stages(trainedModel.stages.size -1).asInstanceOf[ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel]

// 2. get the nativeBooster in order to call the feature score getter
val modelNativeForFeatureImportance = model.nativeBooster

// 3. feature score getter
val featureScore = modelNativeForFeatureImportance.getFeatureScore()

val featureScoreDf = featureScore.toSeq.toDF("feature", "score")

// COMMAND ----------

display(featureScoreDf)

// COMMAND ----------

// DBTITLE 1,Feature Importance #2
// create a list of all relevant features 
val listOfColsForFeatureImportance = updatedAfterModel.drop("label",
                               "dummyString",
                               "dummyString_indexed",
                               "features",
                               "prediction").columns.toSeq

val mapOfCols = listOfColsForFeatureImportance.map(t => t.toString() -> ("f"))
var colsFinalMap = scala.collection.mutable.Map[String, String]()

var i = 0
for ((k,v) <- mapOfCols){
  val updatedStr = v + i.toString
  colsFinalMap += (k -> updatedStr)
  i += 1
}

// COMMAND ----------

// DBTITLE 1,Feature Importance #3
val scoredDfFeatures = featureScoreDf.sort("feature")
scoredDfFeatures.createOrReplaceTempView("scoreDf")

val mapDf = colsFinalMap.toSeq.toDF("Feature", "FeatureID")

mapDf.createOrReplaceTempView("mapDf")

// COMMAND ----------

// DBTITLE 1,Feature Importance #4 (with Information Gain)
// MAGIC %sql
// MAGIC SELECT a.Feature, b.score 
// MAGIC FROM
// MAGIC mapDf a
// MAGIC INNER JOIN scoreDf b
// MAGIC ON a.FeatureID = b.feature
// MAGIC ORDER BY b.score DESC