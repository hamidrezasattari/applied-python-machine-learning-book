from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
import os,sys


os.environ['PYSPARK_PYTHON'] = sys.executable
spark = SparkSession \
    .builder \
    .appName("Spark SQL ETL Maping") \
     .getOrCreate()
sourceDF = spark.createDataFrame([(1,"Alice", 10,3,"female"),(2,"Bob", 44,7,"male")], ["user_id","name", "age","grade","gender"])
sourceDF.collect
sourceDF.show()
sourceDF.createOrReplaceTempView("people")
young = spark.sql("SELECT name FROM people WHERE age < 35")
young.show()