from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
import os,sys


os.environ['PYSPARK_PYTHON'] = sys.executable
spark = SparkSession \
    .builder \
    .appName("Spark SQL ETL Maping") \
     .getOrCreate()
sourceDF = spark.createDataFrame([(1,"Alice", 28,"female",36000),(2,"Bob", 44,"male",48000),(3,"Allen", 34,"male",58000),(4,"Lisa", 21,"female",25000),(5,"Anna", 31,"female",34000)], ["user_id","name", "age","gender","salary"])
sourceDF.collect
sourceDF.show()
targetDF=sourceDF.select("user_id","age","gender","salary").filter(sourceDF.age > 25)      .groupBy( "gender").agg({"salary": "avg", "age": "max"})
targetDF.show()


