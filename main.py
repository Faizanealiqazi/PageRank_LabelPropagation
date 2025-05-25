import os
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, explode, size, lit
from rename_spark_files import rename_single_csv
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")

def run_pagerank(df, iterations=10, damping=0.85):
    links = df.groupBy("src").agg(collect_list("dst").alias("neighbors"))
    ranks = links.select(col("src"), lit(1.0).alias("rank"))

    for _ in range(iterations):
        joined = links.join(ranks, "src")
        exploded = joined.select(
            explode("neighbors").alias("dst"),
            (col("rank") / size(col("neighbors"))).alias("contrib")
        )
        contribs = exploded.groupBy("dst").sum("contrib").withColumnRenamed("sum(contrib)", "contrib")
        ranks = contribs.select(
            col("dst").alias("src"),
            ((1 - damping) + damping * col("contrib")).alias("rank")
        )

    return ranks

def run_label_propagation(edges_df, max_iter=5):
    nodes = edges_df.select("src").union(edges_df.select("dst")).distinct().withColumnRenamed("src", "id")
    labeled = nodes.withColumn("label", col("id"))

    for _ in range(max_iter):
        joined = edges_df.join(labeled, edges_df.dst == labeled.id, how="left").select(edges_df.src.alias("id"), "label")
        label_counts = joined.groupBy("id", "label").count()
        max_labels = label_counts.groupBy("id").agg({"count": "max"}).withColumnRenamed("max(count)", "max_count")
        max_labels = max_labels.join(label_counts, ["id"])
        labeled = max_labels.dropDuplicates(["id"])

    return labeled

def create_spark_session():
    spark = SparkSession.builder \
        .appName("PageRankLabelPropagation") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.memoryOverhead", "2g") \
        .config("spark.sql.inMemoryColumnarStorage.spill.enabled", "true") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.shuffle.manager", "tungsten-sort") \
        .config("spark.shuffle.spill", "true") \
        .getOrCreate()

    # Set log level to WARN to reduce info and warning logs
    spark.sparkContext.setLogLevel("WARN")
    return spark

def explore_data(spark, data_path):
    df = spark.read.csv(data_path, sep='\t', inferSchema=True)
    df = df.withColumnRenamed('_c0', 'src').withColumnRenamed('_c1', 'dst')

    print("Sample data:")
    df.show(5)

    print("Total edges:", df.count())
    print("Unique nodes:", df.select("src").union(df.select("dst")).distinct().count())
    return df

if __name__ == "__main__":
    spark = create_spark_session()
    df = explore_data(spark, DATA_PATH)

    print("Running PageRank...")
    pagerank_result = run_pagerank(df)
    pagerank_result.show(5)

    # Save PageRank result to a single CSV file
    pagerank_result.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/pagerank_result")
    rename_single_csv("output/pagerank_result", "pagerank_result")

    print("Running Label Propagation...")
    labelprop_result = run_label_propagation(df)
    labelprop_result.show(5)

    # Save Label Propagation result to a single CSV file
    labelprop_result.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/labelprop_result")
    rename_single_csv("output/labelprop_result", "labelprop_result")

    spark.stop()


