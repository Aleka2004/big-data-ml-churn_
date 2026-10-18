from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession, functions as F


def build_dataset(spark: SparkSession, rows: int, seed: int = 42):
    """Spark ішінде сыртқы файлсыз синтетикалық customer churn деректерін жасайды."""
    df = spark.range(rows).withColumnRenamed("id", "customer_id")

    df = (
        df
        .withColumn("age", (F.floor(F.rand(seed) * 57) + 18).cast("int"))
        .withColumn("tenure_months", (F.floor(F.rand(seed + 1) * 72) + 1).cast("int"))
        .withColumn("monthly_spend", F.round(F.greatest(F.lit(5.0), F.least(F.lit(180.0), 55 + 22 * F.randn(seed + 2))), 2))
        .withColumn("usage_hours", F.round(F.greatest(F.lit(1.0), F.least(F.lit(120.0), 42 + 18 * F.randn(seed + 3))), 2))
        .withColumn("support_calls", F.floor(F.rand(seed + 4) * 6).cast("int"))
        .withColumn("late_payments", F.floor(F.rand(seed + 5) * 5).cast("int"))
        .withColumn("satisfaction_score", F.round(1 + 4 * F.rand(seed + 6), 2))
        .withColumn(
            "contract_type",
            F.when(F.rand(seed + 7) < 0.55, F.lit("month-to-month"))
            .when(F.rand(seed + 7) < 0.83, F.lit("one-year"))
            .otherwise(F.lit("two-year")),
        )
        .withColumn(
            "region",
            F.when(F.rand(seed + 8) < 0.32, F.lit("Almaty"))
            .when(F.rand(seed + 8) < 0.57, F.lit("Astana"))
            .when(F.rand(seed + 8) < 0.73, F.lit("Shymkent"))
            .otherwise(F.lit("Other")),
        )
        .withColumn("has_internet", F.when(F.rand(seed + 9) < 0.88, F.lit(1)).otherwise(F.lit(0)))
    )

    risk = (
        F.lit(-1.5)
        + 0.55 * F.when(F.col("contract_type") == "month-to-month", 1.0).otherwise(0.0)
        - 0.55 * F.when(F.col("contract_type") == "two-year", 1.0).otherwise(0.0)
        - 0.022 * F.col("tenure_months")
        + 0.18 * F.col("support_calls")
        + 0.28 * F.col("late_payments")
        - 0.75 * (F.col("satisfaction_score") - 3.0)
        + 0.009 * (F.col("monthly_spend") - 55.0)
        - 0.015 * (F.col("usage_hours") - 42.0)
        + 0.25 * (1 - F.col("has_internet"))
        + 0.90 * F.when((F.col("support_calls") >= 3) & (F.col("satisfaction_score") < 3), 1.0).otherwise(0.0)
        + 0.70 * F.when((F.col("monthly_spend") > 80) & (F.col("contract_type") == "month-to-month"), 1.0).otherwise(0.0)
        + 0.65 * F.when(F.col("late_payments") >= 2, 1.0).otherwise(0.0)
        + 0.35 * F.randn(seed + 10)
    )
    prob = 1.0 / (1.0 + F.exp(-risk))
    df = df.withColumn("churn", F.when(F.rand(seed + 11) < prob, 1.0).otherwise(0.0))
    return df.cache()


def positive_class_metrics(predictions):
    stats = predictions.select(
        F.sum(F.when((F.col("label") == 1) & (F.col("prediction") == 1), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col("label") == 0) & (F.col("prediction") == 1), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("label") == 1) & (F.col("prediction") == 0), 1).otherwise(0)).alias("fn"),
    ).first()
    tp, fp, fn = float(stats.tp), float(stats.fp), float(stats.fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Big Data ML: customer churn with Apache Spark MLlib")
    parser.add_argument("--rows", type=int, default=1_000_000, help="Синтетикалық жазба саны")
    parser.add_argument("--output", default="results/spark_metrics", help="Нәтиже сақталатын бума")
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("BigDataMLChurn")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = build_dataset(spark, args.rows)
    print(f"Rows: {df.count():,}")
    df.groupBy("churn").count().orderBy("churn").show()

    categorical = ["contract_type", "region"]
    numeric = [
        "age", "tenure_months", "monthly_spend", "usage_hours",
        "support_calls", "late_payments", "satisfaction_score", "has_internet",
    ]

    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in categorical]
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in categorical],
        outputCols=[f"{c}_ohe" for c in categorical],
    )
    assembler = VectorAssembler(
        inputCols=numeric + [f"{c}_ohe" for c in categorical],
        outputCol="features",
        handleInvalid="keep",
    )
    data = df.withColumnRenamed("churn", "label")
    train, test = data.randomSplit([0.8, 0.2], seed=42)

    # Класс дисбалансын азайту үшін train жиынында салмақ есептейміз.
    counts = {float(r["label"]): int(r["count"]) for r in train.groupBy("label").count().collect()}
    total = sum(counts.values())
    w0 = total / (2.0 * counts.get(0.0, 1))
    w1 = total / (2.0 * counts.get(1.0, 1))
    train = train.withColumn("class_weight", F.when(F.col("label") == 1.0, F.lit(w1)).otherwise(F.lit(w0)))

    models = {
        "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="label", weightCol="class_weight", maxIter=80, regParam=0.02),
        "Random Forest": RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="class_weight", numTrees=120, maxDepth=10, seed=42),
        "Gradient Boosted Trees": GBTClassifier(featuresCol="features", labelCol="label", weightCol="class_weight", maxIter=80, maxDepth=6, stepSize=0.08, seed=42),
    }

    auc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

    rows = []
    for name, estimator in models.items():
        pipeline = Pipeline(stages=indexers + [encoder, assembler, estimator])
        model = pipeline.fit(train)
        pred = model.transform(test).cache()

        auc = auc_eval.evaluate(pred)
        acc = acc_eval.evaluate(pred)
        precision, recall, f1 = positive_class_metrics(pred)
        rows.append((name, float(acc), precision, recall, f1, float(auc)))
        print(f"{name}: accuracy={acc:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, AUC={auc:.4f}")
        pred.unpersist()

    result = spark.createDataFrame(rows, ["model", "accuracy", "precision", "recall", "f1", "auc"])
    result.orderBy(F.desc("auc")).show(truncate=False)
    result.coalesce(1).write.mode("overwrite").option("header", True).csv(args.output)

    spark.stop()


if __name__ == "__main__":
    main()

