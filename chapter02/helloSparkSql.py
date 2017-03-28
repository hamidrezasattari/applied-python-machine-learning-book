from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .master("local[*]") \
    .getOrCreate()

dfTblUser= spark.read \
      .format("jdbc") \
      .option("url", "jdbc:mysql://localhost:3306/test") \
      .option("driver", "com.mysql.jdbc.Driver")\
      .option("dbtable", "users")\
      .option("user", "root")\
      .option("password", "password")\
      .load()

dfTblUser.show();



