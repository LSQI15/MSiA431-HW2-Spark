from pyspark.sql import SQLContext
from pyspark import SparkContext
import matplotlib.pyplot as plt

sc = SparkContext()
sqlcontext = SQLContext(sc)
path = "hdfs://wolf.analytics.private/user/slx4192/data/crime/Crimes_-_2001_to_present.csv"
mydata = sqlcontext.read.csv(path, header=True)
mydata_month = mydata\
                .withColumn('Month', mydata['Date'].substr(0, 2))
monthly_avg = mydata_month\
    .groupBy("Month", "Year")\
    .count()\
    .select("Month", "count")\
    .groupBy("Month")\
    .avg()\
    .orderBy("Month")
monthly_avg.toPandas().plot.bar(x="Month", y="avg(count)")
plt.savefig("slx4192_1.png")