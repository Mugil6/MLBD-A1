import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, col, concat_ws, collect_list, regexp_replace, lower, udf, desc
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, Normalizer
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("Q11_Similarity").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Dynamic Path
base_path = os.environ.get("GUTENBERG_ROOT")
if not base_path:
    print("Error: GUTENBERG_ROOT environment variable not set.")
    sys.exit(1)

input_path = f"file://{base_path}/*.txt"

#  Load Bible + 99 Random Books because full cartesian product exceeds local single node RAM---
df_all = spark.read.text(input_path).withColumn("file_name", input_file_name())
target_book = df_all.filter(col("file_name").contains("10.txt"))
other_books = df_all.filter(~col("file_name").contains("10.txt")).limit(99)
df = target_book.union(other_books)
# ---------------------------------------------

# Preprocessing
full_text_df = df.groupBy("file_name").agg(concat_ws(" ", collect_list("value")).alias("text"))
clean_df = full_text_df.withColumn("clean_text", regexp_replace(lower(col("text")), r"[^a-z\s]", ""))

# TF-IDF Pipeline
tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])
model = pipeline.fit(clean_df)
tfidf_df = model.transform(clean_df)

# Cosine Similarity
normalizer = Normalizer(inputCol="features", outputCol="normFeatures")
norm_df = normalizer.transform(tfidf_df).select("file_name", "normFeatures")

similarity_df = norm_df.alias("a").crossJoin(norm_df.alias("b")) \
    .filter(col("a.file_name") != col("b.file_name")) \
    .withColumn("similarity", udf(lambda x, y: float(x.dot(y)), "double")("a.normFeatures", "b.normFeatures"))

print("\n--- Top 5 Books Similar to '10.txt' (Sampled Run due to Hardware Limit) ---")
similarity_df.filter(col("a.file_name").contains("10.txt")) \
             .orderBy(desc("similarity")) \
             .select(col("b.file_name"), col("similarity")) \
             .show(5, truncate=False)

spark.stop()
