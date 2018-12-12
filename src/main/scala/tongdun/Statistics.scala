package tongdun

import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

import scala.collection.mutable

object Statistics {

  case class MyInstance(label: Double, weight: Double, features: Vector)

  class MultiClassSummarizer extends Serializable {
    private val distinctMap = new mutable.HashMap[Int, (Long, Double)]
    private var totalInvalidCnt: Long = 0L

    /**
      * Add a new label into this MultilabelSummarizer, and update the distinct map.
      *
      * @param label  The label for this data point.
      * @param weight The weight of this instances.
      * @return This MultilabelSummarizer
      */
    def add(label: Double, weight: Double = 1.0): this.type = {
      require(weight >= 0.0, s"instance weight, $weight has to be >= 0.0")

      if (weight == 0.0) return this
      //这里要求label必须为整数，否则认为非法
      if (label - label.toInt != 0.0 || label < 0) {
        totalInvalidCnt += 1
        this
      } else {
        val (counts: Long, weightSum: Double) = distinctMap.getOrElse(label.toInt, (0L, 0.0))
        //累加样本次数及weight
        distinctMap.put(label.toInt, (counts + 1L, weightSum + weight))
        this
      }
    }

    /**
      * Merge another MultilabelSummarizer, and update the distinct map.
      * (Note that it will merge the smaller distinct map into the larger one using in-place
      * merging, so either `this` or `other` object will be modified and returned.)
      *
      * @param other The other MultilabelSummarizer to be merged.
      * @return Merged MultilabelSummarizer object.
      */
    def merge(other: MultiClassSummarizer): MultiClassSummarizer = {
      //将size小的并入大的，性能
      val (largeMap, smallMap) = if (this.distinctMap.size > other.distinctMap.size) {
        (this, other)
      } else {
        (other, this)
      }
      smallMap.distinctMap.foreach {
        case (key, value) =>
          val (counts: Long, weightSum: Double) = largeMap.distinctMap.getOrElse(key, (0L, 0.0))
          //直接累加
          largeMap.distinctMap.put(key, (counts + value._1, weightSum + value._2))
      }
      largeMap.totalInvalidCnt += smallMap.totalInvalidCnt
      largeMap
    }

    def numClasses: Int = if (distinctMap.isEmpty) 0 else distinctMap.keySet.max + 1

    def histogram: Array[Double] = {
      val result = Array.ofDim[Double](numClasses)
      var i = 0
      //应该是val len = numClasses
      val len = result.length
      //这里要求class从0到k-1
      while (i < len) {
        result(i) = distinctMap.getOrElse(i, (0L, 0.0))._2
        i += 1
      }
      result
    }

    def countInvalid: Long = totalInvalidCnt

  }

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

    //    rdd.map不能使用vector
    val instances: RDD[MyInstance] =
      rawInput.withColumn("weight", lit(1.0)).select("label", "weight", "sepal_length", "sepal_width", "petal_length", "petal_width").rdd.map {
        case Row(label: Double, weight: Double, v1: Double, v2: Double, v3: Double, v4: Double) =>
          MyInstance(label, weight, Vectors.dense(Array(v1, v2, v3, v4)))
      }
    println(instances.count())
    println(instances.first())

    val (summarizer, labelSummarizer) = {
      val seqOp = (c: (MultivariateOnlineSummarizer, MultiClassSummarizer), instance: MyInstance) =>
        (c._1.add(instance.features), c._2.add(instance.label, instance.weight))

      val combOp = (c1: (MultivariateOnlineSummarizer, MultiClassSummarizer),
                    c2: (MultivariateOnlineSummarizer, MultiClassSummarizer)) =>
        (c1._1.merge(c2._1), c1._2.merge(c2._2))

      instances.treeAggregate(
        (new MultivariateOnlineSummarizer, new MultiClassSummarizer)
      )(seqOp, combOp, 2)
    }

    val histogram = labelSummarizer.histogram
    val numInvalid = labelSummarizer.countInvalid
    val numFeatures = summarizer.mean.size
    println(histogram)
    histogram.foreach(println)
    println(numFeatures)
    println(numInvalid)
  }


}
