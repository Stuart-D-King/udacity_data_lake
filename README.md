## Udacity Data Engineer Nanodegree
## Data Lake
### Introduction and Project Purpose
A music streaming startup, Sparkify, has grown its user base and song database and wants to move its data warehouse to a data lake. The company's data resides in S3, in a directory of JSON logs of user activity on the app, as well as a directory with JSON metadata of the songs in its app.

The company needs an ETL pipeline that extracts its data from S3, processes the data using Spark, and loads the data back into S3 as a set of fact and dimensional tables. This will allow the company's analytics team to continue finding insights in what songs its users are listening to.

The project will require skills in using Spark and AWS services to build an ETL pipeline and data lake hosted on S3. To complete the project, data will be loaded from S3, processed into analytics tables using Spark, and loaded back into S3. The Spark process will be deployed on an AWS EMR cluster.

### Datasets
There are two datasets that reside on S3 that will be loaded into staging tables on Redshift.

- Song data: `s3://udacity-dend/song_data`
- Log data: `s3://udacity-dend/log_data`  

The **song dataset** is a subset of real data from the [Million Song Dataset](http://millionsongdataset.com/). Each file is in JSON format and contains metadata about a song and the artist of that song. The files are partitioned by the first three letters of each song's track ID. For example, here are filepaths to two files in this dataset.

    song_data/A/B/C/TRABCEI128F424C983.json
    song_data/A/A/B/TRAABJL12903CDCF1A.json

Here is an example of what a single song file, TRAABJL12903CDCF1A.json, looks like.

    {"num_songs": 1, "artist_id": "ARJIE2Y1187B994AB7", "artist_latitude": null, "artist_longitude": null, "artist_location": "", "artist_name": "Line Renaud", "song_id": "SOUPIRU12A6D4FA1E1", "title": "Der Kleine Dompfaff", "duration": 152.92036, "year": 0}

The **log dataset** consists of log files in JSON format generated by this [event simulator](https://github.com/Interana/eventsim) based on the songs in the dataset above. These simulate activity logs from a music streaming app based on specified configurations.

The log files in the dataset are partitioned by year and month. For example, here are filepaths to two files in this dataset.

    log_data/2018/11/2018-11-12-events.json
    log_data/2018/11/2018-11-13-events.json

And below is an example of what the data in a log file, 2018-11-12-events.json, looks like.

![log-data](img/log-data.png)

### Data Processing
As data is loaded from S3, the **schema-on-read** approach is utilized to structure the data using Spark. The data is then further processed using the PySpark API, extracting only the data fields needed to build our data model. Once the data is fully processed into individual analytics tables, the data is saved back to S3 as partitioned `parquet` files. The data can then be loaded back into Spark dataframe objects on demand for the Sparkify analytics team to easily query data and quickly output aggregations. For the saved analytics tables, a **star schema** is appropriate given the simplicity of the data model and the presence of one fact table with accompanying dimension tables.

#### Fact Table
**songplays** - records in log data associated with song plays i.e. records with page `NextSong`

| column name | data type |
| ----------- | --------- |
| songplay_id | integer   |
| start_time  | timestamp |
| user_id     | integer   |
| level       | string    |
| song_id     | string    |
| artist_id   | string    |
| session_id  | integer   |
| location    | string    |
| user_agent  | string    |


#### Dimension Tables  
**users** - users in the app

| column name | data type |
| ----------- | --------- |
| user_id     | integer   |
| first_name  | string    |
| last_name   | string    |
| gender      | string    |
| level       | string    |


**songs** - songs in music database  

| column name | data type |
| ----------- | --------- |
| song_id     | string    |
| title       | string    |
| artist_id   | string    |
| year        | integer   |
| duration    | double    |


**artists** - artists in music database   

| column name | data type |
| ----------- | --------- |
| artist_id   | string    |
| name        | string    |
| location    | string    |
| latitude    | double    |
| longitude   | double    |


**time** - timestamps of records in `songplays` broken down into specific units

| column name | data type |
| ----------- | --------- |
| start_time  | timestamp |
| hour        | integer   |
| day         | integer   |
| week        | integer   |
| month       | integer   |
| year        | integer   |
| weekday     | integer   |

### Spark and AWS EMR
The data pipeline is enclosed within the `etl.py` Python script. Steps within the workflow include creating a Spark session object, ingesting song and log files, processing data into individual analytics tables, and saving processed data to S3. If Spark is installed locally and configured to run in standalone mode, the script can be run from the command line:

    python path/to/etl.py

 Initial testing was done locally, and once the script ran successfully on a sample dataset, an AWS EMR cluster was launched to process the complete data files. A Jupyter Notebook on the cluster helped resolve lingering bugs and syntax errors before the `etl.py` Spark application was transferred onto the EMR host and the application was run using `spark-submit` from the command line:

    /usr/bin/spark-submit --master yarn path/to/etl.py


### Conclusion
Transitioning Sparkify's data storage and analytical engine to the Cloud positions the company for long-term data management success. The power and scalability of AWS will allow the company to easily increase or reduce infrastructure and processing power needs as the business grows and analytical needs are refined. Furthermore, building a data lake with Spark and AWS services equips the company with the latest big data tools and technologies; expands its ability to leverage all types of data formats and values, not just tabular and high-value data; and opens the door for even more advanced analytics such as machine learning.

### Example Queries
The number of song plays by hour to understand when during the day users are using the app more frequently - app usage appears to peak between 3-6pm.

    songplays_table.join(time_table, on=[songplays_table.start_time == time_table.start_time]) \
        .groupBy('hour') \
        .agg(f.countDistinct('songplay_id').alias('song_plays')) \
        .orderBy('hour') \
        .show()

The number of song plays by subscription level - paid subscribers are more active.

    songplays_table.groupBy('level') \
        .agg(f.count('songplay_id').alias('song_plays')) \
        .show()

The count of users by gender - there are more female users than male.

    users_table.groupBy('gender') \
        .agg(f.countDistinct('user_id').alias('count')) \
        .show()
