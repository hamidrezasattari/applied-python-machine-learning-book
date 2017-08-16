import os,sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


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

(trainingData, testingData) = data.randomSplit([0.9, 0.1])
trainingData = trainingData.withColumnRenamed("sales","label")



assembler = VectorAssembler(inputCols=["TV","Radio","Newspaper"], outputCol="features")

standardizer = StandardScaler(withMean=True, withStd=True,
                              inputCol='features',
                              outputCol='std_features')


lr = LinearRegression(featuresCol = 'std_features', labelCol = 'label')

pipeline = Pipeline(stages=[assembler,standardizer, lr])

lrModel=pipeline.fit(trainingData);
# Variable's coefficients  and intercept of linear regression
print("Coefficients: %s" % str(lrModel.stages[2].coefficients))
print("Intercept: %s" % str(lrModel.stages[2].intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.stages[2].summary
print("numIterations: %d" % trainingSummary.totalIterations)
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

lrPredictions=lrModel.transform(testingData);
lrPredictions.select("prediction", "sales", "std_features").show(5)

paramGrid = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.fitIntercept, [False, True])\
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
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
cvPrediction.select("prediction", "sales", "std_features").show(5)

