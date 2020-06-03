from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql.functions import *
from pyspark.sql.types import TimestampType
from datetime import datetime
import matplotlib.pyplot as plt

sc = SparkContext()
sqlcontext = SQLContext(sc)
path = "hdfs://wolf.analytics.private/user/slx4192/data/crime/Crimes_-_2001_to_present.csv"
mydata = sqlcontext.read.csv(path, header=True)
mydata = mydata.filter(mydata.Arrest == "true")

# Question4: Find patterns of crimes with arrest with respect to time of the day, day of the week, and month.
# Use whatever method in spark you would like. (25 pts)

########################################################################
# 1. Time of the day
########################################################################
getDateTime = udf(lambda x: datetime.strptime( x, '%m/%d/%Y %I:%M:%S %p'), TimestampType())
mydata_daytime = mydata\
                    .withColumn('Date_time', getDateTime(col('Date')))\
                    .withColumn("Hour", hour(col("Date_time")))\
                    .withColumn("Newdate", to_date(mydata['Date'], "MM/dd/yyyy"))
hourly_avg = mydata_daytime\
                .groupBy("Hour", "Newdate")\
                .count()\
                .select("Hour", "count")\
                .groupBy("Hour")\
                .avg("count")\
                .orderBy("Hour")
hourly_avg.toPandas().plot.bar(x="Hour", y="avg(count)")
plt.title('Average Number of Crime with Arrest by Hour of the Day')
plt.savefig("slx4192_4_Hour_of_Day.png")

########################################################################
# 2. Day of the week
########################################################################
temp = mydata.withColumn("newdate", to_date(mydata['Date'], "MM/dd/yyyy"))
mydata_day_of_week = temp\
                    .withColumn("Day_of_week_number",date_format(temp["newdate"], "u"))\
                    .withColumn("Day_of_week",date_format(temp["newdate"], "E"))
day_of_week_avg = mydata_day_of_week\
                    .groupBy("Day_of_week","Day_of_week_number", "newdate")\
                    .count()\
                    .select("Day_of_week", "Day_of_week_number","count")\
                    .groupBy("Day_of_week", "Day_of_week_number")\
                    .avg()\
                    .orderBy("Day_of_week_number")\
                    .select("Day_of_week", "avg(count)")
day_of_week_avg.toPandas().plot.bar(x="Day_of_week", y="avg(count)")
plt.title('Average Number of Crime with Arrest by Day_of_week')
plt.savefig("slx4192_4_Day_of_Week.png")

########################################################################
# 3. Month of Year
########################################################################
mydata_month = mydata.withColumn('Month', mydata['Date'].substr(0, 2))
monthly_avg = mydata_month\
                .groupBy("Month", "Year")\
                .count()\
                .select("Month", "count")\
                .groupBy("Month")\
                .avg()\
                .orderBy("Month")
monthly_avg.toPandas().plot.bar(x="Month", y="avg(count)")
plt.title('Average Number of Crime with Arrest by Month')
plt.savefig("slx4192_4_Month_of_Year.png")


