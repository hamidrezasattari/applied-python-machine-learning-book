import os,sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.regression import DecisionTreeRegressor



os.environ["PYSPARK_PYTHON"] = "python2"
os.environ['PYTHONPATH'] = ':'.join(sys.path)

spark = SparkSession \
    .builder \
    .appName("Spark ML App") \
     .getOrCreate()
data = spark.read.format("csv")\
.options(header='true', inferschema='true')\
.load("resources/advertising.csv")
data.describe().toPandas().transpose()
data.printSchema()
print(data.describe().toPandas().transpose())
data = data.withColumnRenamed("sales","label")

(trainingData, testingData) = data.randomSplit([0.9, 0.1])



assembler = VectorAssembler(inputCols=["TV","Radio","Newspaper"], outputCol="features")

standardizer = StandardScaler(withMean=True, withStd=True,
                              inputCol='features',
                              outputCol='std_features')


dt = DecisionTreeRegressor(featuresCol="std_features",labelCol = 'label')


pipeline = Pipeline(stages=[assembler,standardizer, dt])

dtModel=pipeline.fit(trainingData);
dtPredictions=dtModel.transform(testingData);
dtPredictions.select("prediction", "label", "std_features").show(5)




lrPredictions=dtModel.transform(testingData);
lrPredictions.select("prediction", "label", "std_features").show(5)

evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(lrPredictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

treeModel = dtModel.stages[2]
# summary only
print(treeModel)

paramGrid = ParamGridBuilder()\
    .addGrid(dt.maxDepth, [2,3,4,5,6,7]) \
    .build()
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(predictionCol='prediction', labelCol='label',metricName= "r2"),
                          numFolds=3)

# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(trainingData)

print(cvModel.avgMetrics)
#print( cvModel.bestModel.stages[2].summary.r2)

for param in paramGrid:
    print param


cvPrediction = cvModel.transform(testingData)
cvPrediction.select("prediction", "label", "std_features").show(5)

