package org.apache.spark.ml.recommendation

import org.apache.spark.ml.linalg.BLAS
import org.apache.spark.sql._
import org.apache.spark.sql.types.{ArrayType, FloatType, IntegerType, StructType}
import org.apache.spark.util.BoundedPriorityQueue


/**
  * Created by wqs on 2019/1/8.
  */
class TDALSModel(override val uid: String,
                 override val rank: Int,
                 @transient override val userFactors: DataFrame,
                 @transient override val itemFactors: DataFrame,
                 val srcBlockSize:Int,
                 val dstBlockSize:Int,
                 val isUserSmall:Boolean,
                 val isItemSmall:Boolean,
                 val enRatio:Boolean,
                 val userPartitionNum:Int,
                 val itemPartitionNum:Int
                )
  extends ALSModel(uid,rank,userFactors,itemFactors){

  override def recommendForAllItems(numUsers: Int): DataFrame = {
    val newUserFactors =
      if(userPartitionNum!=0)
        userFactors.repartition(userPartitionNum)
    else
        userFactors

    val newItemFactors =
    if(itemPartitionNum!=0)
      itemFactors.repartition(itemPartitionNum)
    else
      itemFactors


    recommendForAll(newItemFactors, newUserFactors, $(itemCol), $(userCol), numUsers,
      isItemSmall,isUserSmall)
  }

  override def recommendForAllUsers(numItems: Int): DataFrame = {
    val newUserFactors =
      if(userPartitionNum!=0)
        userFactors.repartition(userPartitionNum)
      else
        userFactors

    val newItemFactors =
      if(itemPartitionNum!=0)
        itemFactors.repartition(itemPartitionNum)
      else
        itemFactors

    recommendForAll(newUserFactors, newItemFactors, $(userCol), $(itemCol), numItems,
      isUserSmall,isItemSmall)
  }


  def recommendForAll(srcFactors: DataFrame,
                dstFactors: DataFrame,
                srcOutputColumn: String,
                dstOutputColumn: String,
                num: Int,
                isSrcFactorsSmall:Boolean,isDstFactorsSmall:Boolean): DataFrame ={
    import userFactors.sparkSession.implicits._

    val srcFactorsBlocked = blockify(srcFactors.as[(Int, Array[Float])],srcBlockSize)
    val dstFactorsBlocked = blockify(dstFactors.as[(Int, Array[Float])],dstBlockSize)



    var ratio:Double = 0
    val tmp = 1000000.0
    if(enRatio){
      val allNum = dstFactors.count()
      ratio = Math.round(num*tmp/allNum)/tmp
    }



    def join(item:Seq[(Int,Array[Float])],
             array:Array[Seq[(Int,Array[Float])]],
             isItemSrc:Boolean): Array[(Seq[(Int,Array[Float])],Seq[(Int,Array[Float])])] ={

      array.map(dst=>{
        if(isItemSrc)
          (item,dst)
        else
          (dst,item)
      })

    }
    val crossJoinDF = if((isSrcFactorsSmall && !isDstFactorsSmall) || (isSrcFactorsSmall && isDstFactorsSmall)) {
      val array = srcFactorsBlocked.collect()
      dstFactorsBlocked.flatMap(item=>join(item,array,false)).as[(Seq[(Int, Array[Float])], Seq[(Int, Array[Float])])]
    }
    else if(!isSrcFactorsSmall && isDstFactorsSmall){
      val array = dstFactorsBlocked.collect()
      srcFactorsBlocked.flatMap(item=>join(item,array,true)).as[(Seq[(Int, Array[Float])], Seq[(Int, Array[Float])])]
    }
    else
      srcFactorsBlocked.crossJoin(dstFactorsBlocked)



     val ratings = crossJoinDF.as[(Seq[(Int, Array[Float])], Seq[(Int, Array[Float])])]
      .flatMap { case (srcIter, dstIter) =>
        val m = srcIter.size
        var firstNum = num
        if(enRatio){
          firstNum = Math.ceil(ratio * dstIter.size).toInt
        }
        val n = math.min(dstIter.size, firstNum)
        val output = new Array[(Int, Int, Float)](m * n)
        var i = 0
        val pq = new BoundedPriorityQueue[(Int, Float)](firstNum)(Ordering.by(_._2))
        srcIter.foreach { case (srcId, srcFactor) =>
          dstIter.foreach { case (dstId, dstFactor) =>
            // We use F2jBLAS which is faster than a call to native BLAS for vector dot product
            val score = BLAS.f2jBLAS.sdot(rank, srcFactor, 1, dstFactor, 1)
            pq += dstId -> score
          }
          pq.foreach { case (dstId, score) =>
            output(i) = (srcId, dstId, score)
            i += 1
          }
          pq.clear()
        }
        output.toSeq
      }
//    // We'll force the IDs to be Int. Unfortunately this converts IDs to Int in the output.
    val topKAggregator = new TopByKeyAggregator[Int, Int, Float](num, Ordering.by(_._2))
    val recs = ratings.as[(Int, Int, Float)].groupByKey(_._1).agg(topKAggregator.toColumn)
      .toDF("id", "recommendations")


    val arrayType = ArrayType(
      new StructType()
        .add(dstOutputColumn, IntegerType)
        .add("rating", FloatType)
    )
    recs.select($"id".as(srcOutputColumn), $"recommendations".cast(arrayType))
  }

  private def blockify(factors: Dataset[(Int, Array[Float])],
                       blockSize: Int = 4096): Dataset[Seq[(Int, Array[Float])]] = {
    import factors.sparkSession.implicits._
    factors.mapPartitions(_.grouped(blockSize))
  }


}


object TDALSModel {


  def load(model:ALSModel,
           userBlockSize : Int= 4096,
           itemBlockSize:Int = 4096,
           isUserSmall:Boolean = false,
           isItemSmall:Boolean = true,
           enRatio:Boolean = true,
           userPartitionNum:Int = 0,
           itemPartitionNum:Int = 0
          ): TDALSModel ={

    val td_asl_model = new TDALSModel(model.uid,model.rank,
      model.userFactors,model.itemFactors,
      srcBlockSize = userBlockSize,
      dstBlockSize = itemBlockSize,
      isUserSmall  = isUserSmall,
      isItemSmall = isItemSmall,
      enRatio = enRatio,
      userPartitionNum,
      itemPartitionNum
    )
    td_asl_model.setUserCol(model.getUserCol)
    td_asl_model.setItemCol(model.getItemCol)
    td_asl_model.setPredictionCol(model.getPredictionCol)
    td_asl_model

  }

  def main(args: Array[String]) {

//    TDALSModel.load("")

  }

}
