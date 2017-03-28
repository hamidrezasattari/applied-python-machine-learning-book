from pyspark.sql import SparkSession
from pyspark.sql.types import *
import os,sys


os.environ['PYSPARK_PYTHON'] = sys.executable

spark = SparkSession \
    .builder \
    .appName("Spark SQL read from file") \
     .getOrCreate()



path1 = "resources/people.txt"
peopleDF = spark.read.text(path1)
peopleDF.printSchema()

schema = StructType([
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True)])

peopleWithSchemaDF = spark.read.schema(schema).text(path1)
peopleWithSchemaDF.printSchema()

tt = spark.createDataFrame([("Alice", 10), ("Bob", 12)], ["name", "age"])
tt.collect()
tt.printSchema()




