import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, col, regexp_extract, first, when, abs, desc

spark = SparkSession.builder.appName("Q12_Network").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# DYNAMIC PATH CONFIGURATION
base_path = os.environ.get("GUTENBERG_ROOT")
if not base_path:
    print("Error: GUTENBERG_ROOT environment variable not set.")
    sys.exit(1)

input_path = f"file://{base_path}/*.txt"
# ----------------------------------

df = spark.read.text(input_path).withColumn("file_name", input_file_name())

meta_raw = df.withColumn("author", regexp_extract(col("value"), r"Author:\s*(.*)", 1)) \
             .withColumn("release_date", regexp_extract(col("value"), r"Release Date:\s*(.*)", 1))

books_df = meta_raw.groupBy("file_name").agg(
    first(when(col("author") != "", col("author")), ignorenulls=True).alias("author"),
    first(when(col("release_date") != "", col("release_date")), ignorenulls=True).alias("release_date")
)

books_df = books_df.withColumn("year", regexp_extract(col("release_date"), r"(\d{4})", 1)) \
                   .filter(col("author").isNotNull() & col("year").isNotNull()) \
                   .select("author", "year").distinct()

X = 5
edges = books_df.alias("a").crossJoin(books_df.alias("b")) \
    .filter(
        (col("a.author") != col("b.author")) & 
        (abs(col("a.year").cast("int") - col("b.year").cast("int")) <= X)
    ) \
    .select(col("a.author").alias("src"), col("b.author").alias("dst"))

print("\n--- Top 5 Influencers (Out-Degree) ---")
edges.groupBy("src").count().withColumnRenamed("count", "out_degree") \
     .orderBy(desc("out_degree")).show(5, truncate=False)

spark.stop()
