package tongdun

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ArrayBuffer

object PmmlModel {
  def main(args: Array[String]): Unit = {
    val inputPath = "hdfs:///user/turing/data/pengyuan.li/data_temp/u-401dbdf2-ec01-4eee-878e-7e5568b9d738-800c1W_tps2.csv"
    val spark = SparkSession
      .builder()
      .appName("Pipeline-Spark Example")
      .getOrCreate()
    val rawInput = spark.read.format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat").option("header", true).option("inferSchema", true).load(inputPath)
    rawInput.show()
    val xCol1000 = ArrayBuffer[String]()
    var cnt = 0
    for (c <- rawInput.columns) {
      cnt = cnt + 1
      if (cnt <= 1000) {
        xCol1000.append(c)
      }
    }

    val yCol = "y"
    val xCols = xCol1000.mkString("+")
    val formula = new RFormula().setFormula(s"""$yCol ~ $xCols""")
    //    测试决策树的树结构
    val tree = new RandomForestClassifier().setFeaturesCol(formula.getFeaturesCol).setLabelCol(formula.getLabelCol)
    val pipeline = new Pipeline().setStages(Array(formula, tree))
    val model = pipeline.fit(rawInput)
    model.write.overwrite().save("hdfs:///user/turing/data/pengyuan.li/data_temp/pipeline_rf")

  }
}
