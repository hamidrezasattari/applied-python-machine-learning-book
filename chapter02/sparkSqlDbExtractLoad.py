from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .master("local[*]") \
    .getOrCreate()

userSqlQuery="( select u.user_id, username,COUNT(address) as cnt from users u,users_info ui where u.user_id=ui.user_id group by user_id) as userSummary"

dfTblUserSummary= spark.read \
      .format("jdbc") \
      .option("url", "jdbc:mysql://localhost:3306/test") \
      .option("driver", "com.mysql.jdbc.Driver")\
      .option("dbtable",userSqlQuery)\
      .option("user", "root")\
      .option("password", "password")\
      .load()

dfTblUserSummary.show();



dfTblUserSummary.select("user_id","username","cnt").write.mode('append') \
    .format("jdbc") \
    .option("url", "jdbc:mysql://localhost:3306/target") \
    .option("dbtable", "user_summary") \
    .option("rewriteBatchedStatements", "true") \
    .option("user", "root") \
    .option("password", "password") \
    .save()




