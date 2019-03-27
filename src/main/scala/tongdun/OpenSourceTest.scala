package tongdun

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.feature.{RFormula, VectorAssembler}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{FMWithLBFGS, FMWithSGD, LabeledPoint}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

object OpenSourceTest {
  def main(args: Array[String]): Unit = {
    val inputPath = "file:///Users/pengyuan.li/Documents/code/xgboost_scala_demo/src/main/resources/Iris.csv"
    val spark = SparkSession
      .builder()
      .master("local[4]")
      .appName("XGBoost4J-Spark Pipeline Example")
      .getOrCreate()

    // Load dataset
//    val schema = new StructType(Array(
//      StructField("sepal_length", DoubleType, true),
//      StructField("sepal_width", DoubleType, true),
//      StructField("petal_length", DoubleType, true),
//      StructField("petal_width", DoubleType, true),
//      StructField("label", DoubleType, true)))
//    val rawInput = spark.read.schema(schema).csv(inputPath)
    val rawInput = spark.sqlContext.read.format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat").option("header",true).option("inferSchema",true).load(inputPath)
    rawInput.show()
    println(rawInput.schema)
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
    //    测试决策树的树结构
    val tree = new DecisionTreeClassifier().setFeaturesCol(formula.getFeaturesCol).setLabelCol(formula.getLabelCol)
    val pipeline = new Pipeline().setStages(Array(formula, tree))
    val model = pipeline.fit(training)
    val prediction = model.transform(test)
    prediction.show()
    println(model.stages.last.asInstanceOf[DecisionTreeClassificationModel].toDebugString)
    //    测试KNN算法
    //    val knn = new KNNClassifier().setK(10).setTopTreeSize(10).setFeaturesCol(formula.getFeaturesCol).setLabelCol(formula.getLabelCol)
    //    val pipeline2 = new Pipeline().setStages(Array(formula, knn))
    //    val model2 = pipeline2.fit(training)
    //    val prediction2 = model2.transform(test)
    //    prediction2.show()


    val training2 = rawInput.select("label", "sepal_length", "sepal_width", "petal_length", "petal_width").rdd.map {
      case Row(label: Double, v1: Double, v2: Double, v3: Double, v4: Double) =>
        LabeledPoint(label, Vectors.dense(Array(v1, v2, v3, v4)))
    }


    val fm1 = FMWithSGD.train(training2, task = 1, numIterations = 100, stepSize = 0.15, miniBatchFraction = 1.0, dim = (true, true, 4), regParam = (0, 0, 0), initStd = 0.1)


    val fm2 = FMWithLBFGS.train(training2, task = 1, numIterations = 20, numCorrections = 5, dim = (true, true, 4), regParam = (0, 0, 0), initStd = 0.1)

    println("hello")

    val parsedData = rawInput.select("sepal_length", "sepal_width", "petal_length", "petal_width").rdd.map {
      case Row(v1: Double, v2: Double, v3: Double, v4: Double) =>
        Vectors.dense(Array(v1, v2, v3, v4))
    }
    import org.apache.spark.mllib.clustering.dbscan.DBSCAN

    val model2 = DBSCAN.train(parsedData, eps = 0.3F, minPoints = 10, maxPointsPerPartition = 250)
    println("dbscan")
    val clustered = model2.labeledPoints
      .map(p => (p, p.cluster))
      .collectAsMap()
    println(clustered)
  }
}
