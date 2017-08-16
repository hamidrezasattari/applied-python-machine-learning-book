import os,sys
print(sys.executable)
from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier
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
layers = [16, 20, 20, 10]

# create the trainer and set its parameters
mlp = MultilayerPerceptronClassifier( layers=layers,  labelCol="label", featuresCol="std_features")

pipeline = Pipeline(stages=[standardizer , mlp])


mlpModel=pipeline.fit(trainingData);
mlpPredictions=mlpModel.transform(testingData);
mlpPredictions.select("prediction", "label", "std_features").show(5)
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(mlpPredictions)

print("Accuracy on test data = %g" % accuracy)

paramGrid = ParamGridBuilder().\
    addGrid(mlp.maxIter, [ 50,100,150]).\
    addGrid(mlp.blockSize, [ 64,128]). \
    addGrid(mlp.layers, [(16, 10, 10, 10),(16, 32, 32, 10)]). \
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

