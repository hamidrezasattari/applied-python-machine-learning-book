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
targetDF=sourceDF.select("user_id","age","grade").withColumnRenamed("user_id","id").withColumnRenamed("grade","grade_level").cache()
sourceDF.show()
targetDF.show()

def ageCategory(age):
    if age >= 60: return 'old'
    elif age >= 50: return 'average'
    elif age >= 35: return 'mid'
    else: return 'young'
udfAgeCat=udf(ageCategory, StringType())
target2DF =sourceDF.select("user_id","age","grade").withColumn("age_category",udfAgeCat("age")).cache()
target2DF.show()
