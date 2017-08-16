import os,sys
print(sys.executable)
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit



os.environ["PYSPARK_PYTHON"] = "python2"
os.environ['PYTHONPATH'] = ':'.join(sys.path)



spark = SparkSession \
    .builder \
    .appName("Spark ML App") \
     .getOrCreate()

schema = StructType([ \
    StructField("userId",  IntegerType()), \
    StructField("movieId", IntegerType()), \
    StructField("rating", FloatType())])

ratings = spark.read.format("csv")\
.options(header='false') \
.option("delimiter","\\t") \
.schema(schema) \
.load("resources/sample_movies_users.data")
print(ratings.describe().toPandas().transpose())

(training, test) = ratings.randomSplit([0.8, 0.2])
als = ALS(rank=10, maxIter=10,userCol='userId', itemCol='movieId', ratingCol='rating', regParam=0.1, coldStartStrategy="drop")
alsModel = als.fit(training)

predictions = alsModel.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))



testDF = spark.createDataFrame(
     [(0, 50, -1), (0, 172, -1), (0, 133, -1)],
   ["userId", "movieId", "rating"])
predictionDF=alsModel.transform(testDF);
predictionDF.show(5)


tvs = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)




spark.stop()



