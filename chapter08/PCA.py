import os,sys
print(sys.executable)
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import PCA


os.environ["PYSPARK_PYTHON"] = "python2"
os.environ['PYTHONPATH'] = ':'.join(sys.path)

spark = SparkSession \
    .builder \
    .appName("Spark ML App") \
     .getOrCreate()
data = spark.read.format("csv")\
.options(header='false', inferschema='true')\
.load("resources/sonar.all-data")
data.printSchema()
data = data.withColumnRenamed("_c60","label")




vectorassembler = VectorAssembler(
    inputCols=['_c%d' % i for i in range(60)],
    outputCol="features")
output = vectorassembler.transform(data)

standardizer = StandardScaler(withMean=True, withStd=True,
                              inputCol='features',
                              outputCol='std_features')
model = standardizer.fit(output)
output = model.transform(output)

indexer = StringIndexer(inputCol="label", outputCol="label_idx")
indexed = indexer.fit(output).transform(output)
sonar = indexed.select(['std_features', 'label', 'label_idx'])
sonar.show(n=5)


pca = PCA(k=3, inputCol="std_features", outputCol="pca")
pcaModel = pca.fit(sonar)
transformed = pcaModel.transform(sonar)
transformed.select("pca").show(truncate=False)





