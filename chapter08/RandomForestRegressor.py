import os,sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.regression import RandomForestRegressor


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


#dt = DecisionTreeRegressor(featuresCol="std_features",labelCol = 'label')
rf = RandomForestRegressor(featuresCol="std_features",labelCol = 'label')

pipeline = Pipeline(stages=[assembler,standardizer, rf])

rfModel=pipeline.fit(trainingData);
dtPredictions=rfModel.transform(testingData);
dtPredictions.select("prediction", "label", "std_features").show(5)




lrPredictions=rfModel.transform(testingData);
lrPredictions.select("prediction", "label", "std_features").show(5)

evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(lrPredictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

randomForestModel = rfModel.stages[2]
# summary only
print(randomForestModel)

paramGrid = ParamGridBuilder()\
    .addGrid(rf.maxDepth, [ 14,18,22]) \
.addGrid(rf.numTrees, [  24,28,30]) \
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

