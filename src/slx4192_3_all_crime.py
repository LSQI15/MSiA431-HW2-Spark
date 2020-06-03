from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql.functions import *
import pyspark.sql.functions as func
from pyspark.sql.types import TimestampType
from datetime import datetime
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler,StringIndexer,OneHotEncoderEstimator
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor

sc = SparkContext()
sqlcontext = SQLContext(sc)
path = "hdfs://wolf.analytics.private/user/slx4192/data/crime/Crimes_-_2001_to_present.csv"
mydata = sqlcontext.read.csv(path, header=True)

getDateTime = udf(lambda x: datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p'), TimestampType())
mydata_violent = mydata\
                    .withColumn('Date_time', getDateTime(mydata.Date))\
                    .withColumn('Week_num', weekofyear('Date_time'))\
                    .withColumn("Violent", func.when(mydata["IUCR"].like("01%") | mydata["IUCR"].like("02%") |\
                                  mydata["IUCR"].like("03%") | mydata["IUCR"].like("04%") |\
                                  mydata["IUCR"].like("05%") | mydata["IUCR"].like("06%") |\
                                  mydata["IUCR"].like("10%") | mydata["IUCR"].like("13%") |\
                                  mydata["IUCR"].like("24%") | mydata["IUCR"].like("39%") |\
                                  mydata["IUCR"].like("42%"), 1).otherwise(0))

########################################################################
# make a dataframe for all crimes (regardless with it was violent or not
########################################################################
agg_crime = mydata_violent.groupBy("Beat", "Year", "Week_num", "Violent").count()
nv_crime = mydata_violent\
            .groupBy("Year", "Beat", "Week_num")\
            .count()\
            .orderBy("Beat", "Year", "Week_num")\
            .withColumn("Year_WeekNum", concat(mydata_violent.Year, lpad(mydata_violent.Week_num, 3, "-0")))\
            .drop("Year", "Week_num")

########################################################################
# Feature Engineering
########################################################################

# create lag1-lag8 as features for training the model
nv_crime_w_lag = nv_crime\
        .withColumn('lag1', lag('count').over(Window.partitionBy("Beat").orderBy("Year_WeekNum")))\
        .withColumn('lag2', lag('count', 2).over(Window.partitionBy("Beat").orderBy("Year_WeekNum")))\
        .withColumn('lag3', lag('count', 3).over(Window.partitionBy("Beat").orderBy("Year_WeekNum")))\
        .withColumn('lag4', lag('count', 4).over(Window.partitionBy("Beat").orderBy("Year_WeekNum")))\
        .withColumn('lag5', lag('count', 5).over(Window.partitionBy("Beat").orderBy("Year_WeekNum")))\
        .withColumn('lag6', lag('count', 6).over(Window.partitionBy("Beat").orderBy("Year_WeekNum")))\
        .withColumn('lag7', lag('count', 7).over(Window.partitionBy("Beat").orderBy("Year_WeekNum")))\
        .withColumn('lag8', lag('count', 8).over(Window.partitionBy("Beat").orderBy("Year_WeekNum")))\
        .orderBy("Beat","Year_WeekNum")

# extract year and month as feature
nv_crime_final = nv_crime_w_lag\
    .withColumn("Year", nv_crime_w_lag["Year_WeekNum"].substr(0,4).cast(IntegerType()))\
    .withColumn("WeekNum", nv_crime_w_lag["Year_WeekNum"].substr(6,2).cast(IntegerType()))\
    .drop("Year_WeekNum").na.drop()

########################################################################
# Model Pipeline
########################################################################
tmp = str(nv_crime_final.select(countDistinct("Beat")).collect()[0])
dis_beats = int(tmp[tmp.find('=')+1:-1])
BeatIdx = StringIndexer(inputCol='Beat', outputCol='BeatIdx')
WeekNumIdx = StringIndexer(inputCol='WeekNum', outputCol='WeekNumIdx')
encoder = OneHotEncoderEstimator(inputCols = ["BeatIdx","WeekNumIdx"], outputCols = ["BeatVec","WeekNumVec"]).setHandleInvalid("keep")
assembler = VectorAssembler(inputCols=["BeatVec", "Year", "WeekNumVec", "lag1", "lag2", "lag3", "lag4", "lag5", "lag6", "lag7", "lag8"], outputCol='features')
gradient_boosted = GBTRegressor(labelCol="count", featuresCol="features", maxBins=dis_beats, maxIter=10)
pipeline = Pipeline(stages=[BeatIdx, WeekNumIdx, encoder, assembler, gradient_boosted])

########################################################################
# Model Training
########################################################################
train, test = nv_crime_final.randomSplit([0.8, 0.2])
model = pipeline.fit(train)
predictions = model.transform(test)

########################################################################
# Model Evaluation
########################################################################
predictions2 = predictions.select(col("count").cast("Float"), col("prediction"))
evaluator_mse = RegressionEvaluator(labelCol="count", predictionCol="prediction", metricName="mse")
mse = evaluator_mse.evaluate(predictions2)
print(mse)

# process the output file for exercise 2
text_file = open("slx4192_3_all_crime.txt", "w")
text_file.write("MSE for predicting the number of all crimes for beats at the week level is : " + str(mse) + "\n")
text_file.close()