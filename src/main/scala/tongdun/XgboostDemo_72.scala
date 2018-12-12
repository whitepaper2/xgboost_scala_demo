package tongdun


import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import ml.dmlc.xgboost4j.scala.spark.XGBoostEstimator
import org.apache.spark.ml.Pipeline
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

object XgboostDemo_72 {
  def main(args: Array[String]): Unit = {
    //    if (args.length != 1) {
    //      println("Usage: SparkMLlibPipeline input_path native_model_path pipeline_model_path")
    //      sys.exit(1)
    //    }

//    val inputPath = args(0)
    val inputPath = "file:///Users/pengyuan.li/Documents/code/xgboost_scala_demo/src/main/resources/Iris.csv"
    val spark = SparkSession
      .builder()
      .master("local[4]")
      .appName("XGBoost4J-Spark Pipeline Example")
      .getOrCreate()
//    val spark = SparkSession.builder.appName("Xgboost-scala").getOrCreate()
//    val spark = SparkSession.builder.master("local[4]").appName("Xgboost-scala").getOrCreate()

    // Load dataset
    val schema = new StructType(Array(
      StructField("sepal_length", DoubleType, true),
      StructField("sepal_width", DoubleType, true),
      StructField("petal_length", DoubleType, true),
      StructField("petal_width", DoubleType, true),
      StructField("label", DoubleType, true)))
    //    val rawInput = spark.read.csv(inputPath)
    val rawInput = spark.read.schema(schema).csv(inputPath)
    rawInput.show()
    // Split training and test dataset
    val Array(training, test) = rawInput.randomSplit(Array(0.8, 0.2), 123)
    println("train cnt = " + training.count())

    val assembler = new VectorAssembler()
      .setInputCols(Array("sepal_length", "sepal_width", "petal_length", "petal_width"))
      .setOutputCol("features")
    val trainingVector = assembler.transform(training)
    val testVector = assembler.transform(test)
    testVector.show()

    val xgbParam = Map("eta" -> 0.8f,
      "max_depth" -> 2,
      "objective" -> "multi:softprob",
      "num_class" -> 3
    )
    val yCol = "label"
    val xCols = "sepal_length+sepal_width+petal_length+petal_width"
    val formula = new RFormula().setFormula(s"""$yCol ~ $xCols""")
    val xgb = new XGBoostEstimator(xgbParam).setFeaturesCol(formula.getFeaturesCol).setLabelCol(formula.getLabelCol)
//    val xgbModel = xgb.train(readData)
    val pipeline = new Pipeline().setStages(Array(formula, xgb))
    val model = pipeline.fit(training)
    val prediction = model.transform(test)

//    val xgb = new XGBoostEstimator(xgbParam)
//    val model = xgb.train(trainingVector)
//    model.write.overwrite().save("file:///Users/pengyuan.li/Documents/code/xgboost_scala_demo/src/main/resources/xgb")
//    val prediction = model.transform(testVector)

    println("model....")
    // Batch prediction
    prediction.show()

    // Model evaluation
    val evaluator = new MulticlassClassificationEvaluator()
    val accuracy = evaluator.evaluate(prediction)
    println("The model accuracy is : " + accuracy)


  }
}
