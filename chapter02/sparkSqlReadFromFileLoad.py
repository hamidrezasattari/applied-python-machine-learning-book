from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Spark SQL read from file") \
    .getOrCreate()

path1 = "resources/employee.json"
employeeDF = spark.read.json(path1)
employeeDF.printSchema()

path2 = "resources/ages.csv"
agesDF = spark.read.option("header", "true").csv(path2)
agesDF.printSchema()

path3 = "resources/people.txt"
peopleDF = spark.read.text(path3)
peopleDF.printSchema()

employeeDF.select("name", "salary").write.mode('overwrite').option("header", "true").save("employee", format="csv")


path4 = "resources/pendigits.t"
pendigitsDF  = spark.read.format("libsvm").load(path4)
pendigitsDF.printSchema()

path5 = "resources/user.data"
userDF = spark.read.option("delimiter", "\\t").csv(path5)
userDF.show(2)
userDF.printSchema()