package tongdun

import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

object Eval2Class {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.master("local[4]").appName("XGBoost4J-Spark Pipeline Example").getOrCreate
    val df = spark.createDataFrame(Seq((0.1, 0.0), (0.1, 1.0), (0.4, 0.0), (0.6, 0.0), (0.6, 1.0), (0.6, 1.0), (0.8, 1.0))).toDF("score","label")
    df.show()
    val predictionAndLabels = df.rdd.map(r=>(r.getDouble(0), r.getDouble(1)))
    // Instantiate metrics object
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)

    // Precision by threshold
    val precision = metrics.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }
  }
}
