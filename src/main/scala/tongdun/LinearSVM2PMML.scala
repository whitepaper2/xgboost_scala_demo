package tongdun

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature.{RFormula, VectorAssembler}
import org.apache.spark.ml.knn.KNN
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{DataType, DoubleType, StructField, StructType}
import tongdun.Statistics.MyInstance
import org.apache.spark.mllib.regression.FMWithLBFGS
import org.apache.spark.mllib.regression.FMWithSGD
import org.dmg.pmml.{DataDictionary, DataField, FieldName, MiningField, MiningSchema, OpType}
import org.dmg.pmml.PMML
import org.dmg.pmml.regression.{RegressionModel,RegressionTable}

object LinearSVM2PMML {
  def main(args: Array[String]): Unit = {
    val inputPath = "file:///Users/pengyuan.li/Documents/code/xgboost_scala_demo/src/main/resources/Iris.csv"
    val spark = SparkSession
      .builder()
      .master("local[4]")
      .appName("XGBoost4J-Spark Pipeline Example")
      .getOrCreate()

    // Load dataset
    val schema = new StructType(Array(
      StructField("sepal_length", DoubleType, true),
      StructField("sepal_width", DoubleType, true),
      StructField("petal_length", DoubleType, true),
      StructField("petal_width", DoubleType, true),
      StructField("label", DoubleType, true)))
    val rawInput = spark.read.schema(schema).csv(inputPath)
    rawInput.show()
    // Split training and test dataset
    val Array(training, test) = rawInput.filter("label in (0, 1)").randomSplit(Array(0.8, 0.2), 123)
    println("train cnt = " + training.count())
    println("test cnt = " + test.count())

    val assembler = new VectorAssembler()
      .setInputCols(Array("sepal_length", "sepal_width", "petal_length", "petal_width"))
      .setOutputCol("features")
    val trainingVector = assembler.transform(training)
    val testVector = assembler.transform(test)
    testVector.show()

    val yCol = "label"
    val xCols = "sepal_length+sepal_width+petal_length+petal_width"
    val formula = new RFormula().setFormula(s"""$yCol ~ $xCols""")
//    线性支持向量机
//    val linearSvc = new LinearSVC().setFeaturesCol(formula.getFeaturesCol).setLabelCol(formula.getLabelCol)
//    val pipeline = new Pipeline().setStages(Array(formula, linearSvc))
//    val model = pipeline.fit(training)
//    val prediction = model.transform(test)
//    prediction.show()
//    println(model.stages.last)

//    import org.apache.spark.mllib.clustering.KMeans
//    // Cluster the data into two classes using KMeans
//    val numClusters = 2
//    val numIterations = 20
//    val parsedData = training.rdd.map{case Row(v1: Double, v2: Double, v3: Double, v4: Double, label: Double) =>
//      Vectors.dense(Array(v1, v2, v3, v4))}
//
//    val clusters = KMeans.train(parsedData, numClusters, numIterations)
//
//    // Export to PMML to a String in PMML format
//    println(s"PMML Model:\n ${clusters.toPMML}")
//
//    // Export the model to a local file in PMML format
//    clusters.toPMML("/Users/pengyuan.li/Documents/code/xgboost_scala_demo/src/main/resources/kmeans.xml")
    import org.apache.spark.ml.clustering.KMeans
    val kModel = new KMeans().setK(2).setFeaturesCol(formula.getFeaturesCol)
    val pipekModel = new Pipeline().setStages(Array(formula, kModel))
    val model = pipekModel.fit(training)
    val prediction = model.transform(test)
    prediction.show()
    val df_schema = training.schema.json
    print(df_schema)
    model.save("file:///Users/pengyuan.li/Documents/code/xgboost_scala_demo/src/main/resources/kmodel")
//    val pmml2 = new PMML().getHeader.setDescription("description")
//    val model = new LinearSVC().fit(trainingVector)
//
//    if (model.coefficients.size > 0) {
//      val fields = new Array[FieldName](model.coefficients.size)
//      val dataDictionary = new DataDictionary
//      val miningSchema = new MiningSchema
//      val regressionTableYES = new RegressionTable(model.intercept).setTargetCategory("1")
//      var interceptNO = 0
//    }

  }

}
