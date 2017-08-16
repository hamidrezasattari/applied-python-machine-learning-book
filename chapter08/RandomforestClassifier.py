import os,sys
print(sys.executable)
from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.feature import StandardScaler


os.environ["PYSPARK_PYTHON"] = "python2"
os.environ['PYTHONPATH'] = ':'.join(sys.path)

spark = SparkSession \
    .builder \
    .appName("Spark ML App") \
     .getOrCreate()
trainingData = spark.read.format("libsvm").load("resources/pendigits")
print(trainingData.describe().toPandas().transpose())


testingData=spark.read.format("libsvm").load("resources/pendigits.t")
trainingData.show(truncate=False)
standardizer = StandardScaler(withMean=True, withStd=True,
                              inputCol='features',
                              outputCol='std_features')



rf = RandomForestClassifier(labelCol="label", featuresCol="std_features")
pipeline = Pipeline(stages=[standardizer, rf])


rfModel=pipeline.fit(trainingData);
rfPredictions=rfModel.transform(testingData);
rfPredictions.select("prediction", "label", "std_features").show(5)
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(rfPredictions)

print("Accuracy on test data = %g" % accuracy)
paramGrid = ParamGridBuilder().\
    addGrid(rf.maxDepth, [ 14,18,24,30]).\
    addGrid(rf.numTrees, [  24,28,30]).\
    build()

tvs = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)

tvsModel = tvs.fit(trainingData)

print(tvsModel.validationMetrics)
for param in paramGrid:
    print param


prediction = tvsModel.transform(testingData)

prediction.show(truncate=False)
