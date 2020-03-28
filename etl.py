# import os
# import configparser
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql import types as t

# AWS credentials are saved locally as environmental variables
# config = configparser.ConfigParser()
# config.read('dl.cfg')
#
# os.environ['AWS_ACCESS_KEY_ID']=config.get('AWS','AWS_ACCESS_KEY_ID')
# os.environ['AWS_SECRET_ACCESS_KEY']=config.get('AWS','AWS_SECRET_ACCESS_KEY')


def create_spark_session():
    '''
    Create and return a SparkSession object
    '''
    spark = SparkSession \
        .builder \
        .config('spark.jars.packages', 'org.apache.hadoop:hadoop-aws:2.7.0') \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    '''
    Read in song data from S3, process the data into song and artist analytics tables, and save output to S3

    INPUT
    spark: SparkSession object
    input_data: file path to data stored on S3
    output_data: file path to where to store data on S3

    OUTPUT
    none
    '''
    # get filepath to song data file
    song_data = input_data + 'song_data/*/*/*/*.json'

    songSchema = t.StructType([
        t.StructField('artist_id', t.StringType()),
        t.StructField('artist_latitude', t.DoubleType()),
        t.StructField('artist_location', t.StringType()),
        t.StructField('artist_longitude', t.DoubleType()),
        t.StructField('artist_name', t.StringType()),
        t.StructField('duration', t.DoubleType()),
        t.StructField('num_songs', t.IntegerType()),
        t.StructField('song_id', t.StringType()),
        t.StructField('title', t.StringType()),
        t.StructField('year', t.IntegerType()),
    ])

    # read song data file
    df = spark.read.json(song_data, schema=songSchema)

    # extract columns to create songs table
    songs_table = df.select('song_id', 'title', 'artist_id', 'year', 'duration').dropDuplicates()

    # write songs table to parquet files partitioned by year and artist
    songs_output = output_data + 'songs'
    songs_table.write.partitionBy('year', 'artist_id').parquet(songs_output, mode='overwrite') # 'error' is default

    # extract columns to create artists table
    artists_table = df.selectExpr('artist_id', 'artist_name as name', 'artist_location as location', 'artist_latitude as latitude', 'artist_longitude as longitude').dropDuplicates()

    # write artists table to parquet files
    artists_output = output_data + 'artists'
    artists_table.write.parquet(artists_output, mode='overwrite') # 'error' is default


def process_log_data(spark, input_data, output_data):
    '''
    Read in log data from S3; process the data into user, time, and song plays analytics tables, and save output to S3

    INPUT
    spark: SparkSession object
    input_data: file path to data stored on S3
    output_data: file path to where to store data on S3

    OUTPUT
    none
    '''
    # get filepath to log data file
    log_data = input_data + 'log_data/2018/11/*.json'

    # read log data file
    df = spark.read.json(log_data)

    # filter by actions for song plays
    df = df.filter(df.page == 'NextSong')

    # extract columns for users table
    users_table = df.selectExpr('userId as user_id', 'firstName as first_name', 'lastName as last_name', 'gender', 'level').dropDuplicates()

    # write users table to parquet files
    users_output = output_data + 'users'
    users_table.write.parquet(users_output, mode='overwrite')

    # create timestamp column from original timestamp column
    get_timestamp = f.udf(lambda x: datetime.fromtimestamp(x/1000), t.TimestampType())

    # create datetime column from original timestamp column
    get_datetime = f.udf(lambda x: datetime.fromtimestamp(x/1000), t.DateType())

    df = df.withColumn('start_time', get_timestamp(f.col('ts')))

    # extract columns to create time table
    time_table = df.select('start_time', \
            f.hour('start_time').alias('hour'), \
            f.dayofmonth('start_time').alias('day'), \
            f.weekofyear('start_time').alias('week'), \
            f.month('start_time').alias('month'), \
            f.year('start_time').alias('year'), \
            f.date_format('start_time', 'u').cast(t.IntegerType()).alias('weekday')).dropDuplicates()

    # write time table to parquet files partitioned by year and month
    time_output = output_data + 'time'
    time_table.write.partitionBy('year', 'month').parquet(time_output, mode='overwrite')

    # read in song data to use for songplays table
    # songs_data = output_data + 'songs/*/*/*.parquet'
    # songs_df = spark.read.parquet(songs_data)

    song_data = input_data + 'song_data/*/*/*/*.json'

    songSchema = t.StructType([
        t.StructField('artist_id', t.StringType()),
        t.StructField('artist_latitude', t.DoubleType()),
        t.StructField('artist_location', t.StringType()),
        t.StructField('artist_longitude', t.DoubleType()),
        t.StructField('artist_name', t.StringType()),
        t.StructField('duration', t.DoubleType()),
        t.StructField('num_songs', t.IntegerType()),
        t.StructField('song_id', t.StringType()),
        t.StructField('title', t.StringType()),
        t.StructField('year', t.IntegerType())
    ])

    # read song data file
    songs_df = spark.read.json(song_data, schema=songSchema)

    songplays_table = df.join(songs_df, on=[df.artist == songs_df.artist_name, df.song == songs_df.title]) \
        .select(df.start_time,
            f.year(df.start_time).alias('year'),
            f.month(df.start_time).alias('month'),
            df.userId.alias('user_id').cast(t.IntegerType()),
            df.level,
            songs_df.song_id,
            songs_df.artist_id,
            df.sessionId.alias('session_id'),
            df.location,
            df.userAgent.alias('user_agent')) \
        .withColumn('songplay_id', f.monotonically_increasing_id())

    # df.createOrReplaceTempView('log_table')
    # songs_df.createOrReplaceTempView('songs_table')
    #
    # # extract columns from joined song and log datasets to create songplays table
    # songplays_table = spark.sql('''
    #     select
    #     l.start_time,
    #     year(l.start_time) as year,
    #     month(l.start_time) as month,
    #     cast(l.userId as int) as user_id,
    #     l.level,
    #     s.song_id,
    #     s.artist_id,
    #     l.sessionId as session_id,
    #     l.location,
    #     l.userAgent as user_agent
    #     from log_table l
    #     join songs_table s on s.artist_name = l.artist and s.title = l.song
    # ''')

    # write songplays table to parquet files partitioned by year and month
    songplays_output = output_data + 'songplays'
    songplays_table.write.partitionBy('year', 'month').parquet(songplays_output, mode='overwrite')


def main():
    '''
    Run ETL process by creating a SparkSession object, creating analytics tables, and saving output to S3
    '''
    spark = create_spark_session()
    input_data = 's3n://udacity-dend/'
    output_data = 's3n://sking-data-engineer/data-lake/parquet-files/'

    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)

    spark.stop()


if __name__ == '__main__':
    main()
