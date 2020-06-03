########################################################################
# Load the data
########################################################################
from pyspark import SparkContext
sc = SparkContext()
myRDD = sc.textFile("hdfs://wolf.analytics.private/user/slx4192/data/crime/Crimes_-_2001_to_present.csv")
header = myRDD.first()
data = myRDD.filter(lambda row: row != header)
splitted_data = data.map(lambda line: line.split(","))

########################################################################
# 1. find the top 10 blocks in crime events in the last 3 years;
########################################################################
top_10_blocks_last_3_years = splitted_data\
    .map(lambda x: (x[3][0:5], x[17]))\
    .filter(lambda x:  x[1] in ["2019", "2018","2017"])\
    .map(lambda x: (x[0], 1))\
    .reduceByKey(lambda a, b: a+b)\
    .sortBy(lambda x: -x[1]).take(10)

########################################################################
# 2. find the two beats that are adjacent with the highest correlation in
# the number of crime events (this will require you looking at the map to
# determine if the correlated beats are adjacent to each other) over the
# last 5 years
########################################################################
beat_as_key = splitted_data\
    .map(lambda x: (x[10], x[17]))\
    .filter(lambda x:  x[1] in ["2019","2018","2017","2016","2015"])\
    .map(lambda x: (x[0]+"-"+x[1],1))\
    .reduceByKey(lambda a, b: a+b)\
    .map(lambda x: (x[0][0:4],x[0][5:], x[1]))\
    .sortBy(lambda x: (x[0], x[1]))\
    .map(lambda x: (x[0], [x[2]]))\
    .reduceByKey(lambda a,b: a+b)
dict = beat_as_key.collectAsMap()
import numpy as np
cor_m = np.corrcoef(list(dict.values()))
beat = list(dict.keys())
results = []
for j in range(len(beat)):
    for i in range(len(beat)):
        if i > j:
            results.append([cor_m[j][i],beat[j],beat[i]])
results.sort(key=lambda x: abs(x[0]), reverse=True)
top_50_correlated_beat = results[:50]

########################################################################
# 3. establish if the number of crime events is different between Majors
# Daly and Emanuel at a granularity of your choice (not only at the city
# level). Find an explanation of results. (20 pts)
########################################################################

# analyze the last five years when they were in charge
Emanuel_year = ['2015', '2016', '2017', '2018', '2019']
Daley_year = ['2006', '2007', '2008', '2009', '2010']

# the average number of arrest in block 0000X during the last five years when
# Emanuel was in charge
Emanuel_avg = splitted_data \
    .map(lambda x: (x[3][0:5], x[8], x[17])) \
    .filter(lambda x: x[1] == 'true') \
    .filter(lambda x: x[0] == "0000X") \
    .filter(lambda x: x[2] in Emanuel_year) \
    .map(lambda x: (x[0], 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .map(lambda x: x[1] / 60).take(1)

# the average number of arrest in block 0000X during the last five years when
# Daley was in charge
Daley_avg = splitted_data \
    .map(lambda x: (x[3][0:5], x[8], x[17])) \
    .filter(lambda x: x[1] == 'true') \
    .filter(lambda x: x[0] == "0000X") \
    .filter(lambda x: x[2] in Daley_year) \
    .map(lambda x: (x[0], 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .map(lambda x: x[1] / 60).take(1)

# process the output file for exercise 2
text_file = open("slx4192_2.txt", "w")
# part 1
text_file.write("Part 1: \n")
text_file.write("Top 10 blocks in terms of crime incidents in the last 3 years \n")
for i in range(len(top_10_blocks_last_3_years)):
    text_file.write(str(top_10_blocks_last_3_years[i]) + "\n")

# part 2
text_file.write("\nPart 2: \n")
text_file.write("Top 50 correlated beat in terms of crime incidents in the last 5 years \n")
text_file.write("Correlation,Beat-1,Beat-2\n")
text_file.write('\n'.join(map(str, results[0:50])))

# part 3
text_file.write("\n\nPart 3: \n")
text_file.write("The average number of arrest in block 0000X during the last five years when Emanuel was in charge is: " + str(Emanuel_avg[0]) + "\n")
text_file.write("The average number of arrest in block 0000X during the last five years when Daley was in charge is: " + str(Daley_avg[0]) + "\n")
text_file.close()

