import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, col, regexp_extract, first, when, desc, length, avg

# Initializing Spark
spark = SparkSession.builder.appName("Q10_Metadata").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

#DYNAMIC PATH CONFIGURATION
base_path = os.environ.get("GUTENBERG_ROOT")
if not base_path:
    print("Error: GUTENBERG_ROOT environment variable not set.")
    sys.exit(1)

input_path = f"file://{base_path}/*.txt"
# ----------------------------------

# Load Data
df = spark.read.text(input_path).withColumn("file_name", input_file_name())

# Extraction Logic
meta_raw = df.withColumn("title", regexp_extract(col("value"), r"Title:\s*(.*)", 1)) \
             .withColumn("release_date", regexp_extract(col("value"), r"Release Date:\s*(.*)", 1)) \
             .withColumn("language", regexp_extract(col("value"), r"Language:\s*(.*)", 1)) \
             .withColumn("encoding", regexp_extract(col("value"), r"Character set encoding:\s*(.*)", 1))

books_df = meta_raw.groupBy("file_name").agg(
    first(when(col("title") != "", col("title")), ignorenulls=True).alias("title"),
    first(when(col("release_date") != "", col("release_date")), ignorenulls=True).alias("release_date"),
    first(when(col("language") != "", col("language")), ignorenulls=True).alias("language"),
    first(when(col("encoding") != "", col("encoding")), ignorenulls=True).alias("encoding")
)

# Analysis
books_df = books_df.withColumn("year", regexp_extract(col("release_date"), r"(\d{4})", 1))

print("\n--- Books Released Per Year ---")
books_df.groupBy("year").count().orderBy("year").show()

print("\n--- Most Common Language ---")
books_df.groupBy("language").count().orderBy(desc("count")).show(1)

print("\n--- Average Title Length ---")
books_df.select(avg(length(col("title")))).show()

spark.stop()
