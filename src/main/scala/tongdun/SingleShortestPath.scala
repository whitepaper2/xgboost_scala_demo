package tongdun

import org.apache.spark.{SparkConf, SparkContext}
import scala.reflect.ClassTag
import org.apache.spark.graphx._
import org.apache.spark.graphx.{VertexId, Edge}
import org.apache.spark.sql.SparkSession

object SingleShortestPath {
  def main(args: Array[String]): Unit = {
    //    val conf = new SparkConf().setAppName("graphx")
    //    val sc = new SparkContext(conf)
    val spark = SparkSession
      .builder()
      .master("local[4]")
      .appName("XGBoost4J-Spark Pipeline Example")
      .getOrCreate()
    val sc = spark.sparkContext
    // vertex && edge
    val vertexArr = Array(
      (1L, ("Alice", 28)),
      (2L, ("Bob", 27)),
      (3L, ("Charlie", 65)),
      (4L, ("David", 42)),
      (5L, ("Ed", 55)),
      (6L, ("Fran", 50))
    )

    val edgeArr = Array(
      Edge(2L, 1L, 7),
      Edge(2L, 4L, 2),
      Edge(3L, 2L, 4),
      Edge(3L, 6L, 3),
      Edge(4L, 1L, 1),
      Edge(5L, 2L, 2),
      Edge(5L, 3L, 8),
      Edge(5L, 6L, 3)
    )

    val vertexRdd = sc.parallelize(vertexArr)
    val edgeRdd = sc.parallelize(edgeArr)
    val graph = Graph(vertexRdd, edgeRdd)
    println("=========================graph property======================")
    graph.vertices.collect().foreach(println)
    graph.vertices.filter {
      case (id, (name, age)) => age > 30
    }.collect().foreach {
      case (id, (name, age)) => println(s"$name : $age")
    }
    graph.edges.collect.foreach(println)


    val myVertices = sc.makeRDD(Array((1L, "A"), (2L, "B"), (3L, "C"), (4L, "D"), (5L, "E"), (6L, "F"), (7L, "G")))
    val initialEdges = sc.makeRDD(Array(Edge(1L, 2L, 7.0), Edge(1L, 4L, 5.0),
      Edge(2L, 3L, 8.0), Edge(2L, 4L, 9.0), Edge(2L, 5L, 7.0),
      Edge(3L, 5L, 5.0),
      Edge(4L, 5L, 15.0), Edge(4L, 6L, 6.0),
      Edge(5L, 6L, 8.0), Edge(5L, 7L, 9.0),
      Edge(6L, 7L, 11.0)))
    val myEdges = initialEdges.filter(e => e.srcId != e.dstId).flatMap(e => Array(e, Edge(e.dstId, e.srcId, e.attr))).distinct()  //去掉自循环边，有向图变为无向图，去除重复边
    val myGraph = Graph(myVertices, myEdges).cache()
    val lpaGraph = graph.mapVertices { case (vid, _) => vid }
    println(lpaGraph)

//    println(dijkstra(myGraph, 3L).vertices.map(x => (x._1, x._2)).collect().mkString(" | "))


  }

  def dijkstra[VD: ClassTag](g : Graph[VD, Double], origin: VertexId) = {
    //初始化，其中属性为（boolean, double，Long）类型，boolean用于标记是否访问过，double为顶点距离原点的距离，Long是上一个顶点的id
    var g2 = g.mapVertices ((vid, _) => (false, if (vid == origin) 0 else Double.MaxValue, - 1L) )

    for (i <- 1L to g.vertices.count () ) {
      //从没有访问过的顶点中找出距离原点最近的点
      val currentVertexId = g2.vertices.filter (! _._2._1).reduce ((a, b) => if (a._2._2 < b._2._2) a else b)._1
      //更新currentVertexId邻接顶点的‘double’值
      val newDistances = g2.aggregateMessages[(Double, Long)] (
        triplet => if (triplet.srcId == currentVertexId && ! triplet.dstAttr._1) {
          //只给未确定的顶点发送消息
          triplet.sendToDst ((triplet.srcAttr._2 + triplet.attr, triplet.srcId) )
        },
        (x, y) => if (x._1 < y._1) x else y,
        TripletFields.All
      )
      //newDistances.foreach(x => println("currentVertexId\t"+currentVertexId+"\t->\t"+x))
      //更新图形
      g2 = g2.outerJoinVertices (newDistances) {
        case (vid, vd, Some (newSum) ) => (vd._1 || vid == currentVertexId, math.min (vd._2, newSum._1), if (vd._2 <= newSum._1) vd._3 else newSum._2)
        case (vid, vd, None) => (vd._1 || vid == currentVertexId, vd._2, vd._3)
      }
      //g2.vertices.foreach(x => println("currentVertexId\t"+currentVertexId+"\t->\t"+x))
    }

    //g2
    g.outerJoinVertices (g2.vertices) ((vid, srcAttr, dist) => (srcAttr, dist.getOrElse (false, Double.MaxValue, - 1)._2, dist.getOrElse (false, Double.MaxValue, - 1)._3) )
  }


  def label_propagation[VD, ED: ClassTag](graph: Graph[VD, ED], maxSteps: Int)={
    val lpaGraph = graph.mapVertices { case (vid, _) => vid }
    def sendMessage(e: EdgeTriplet[VertexId, ED]): Iterator[(VertexId, Map[VertexId, Long])] = {
      Iterator((e.srcId, Map(e.dstAttr -> 1L)), (e.dstId, Map(e.srcAttr -> 1L)))
    }
    def mergeMessage(count1: Map[VertexId, Long], count2: Map[VertexId, Long])
    : Map[VertexId, Long] = {
      (count1.keySet ++ count2.keySet).map { i =>
        val count1Val = count1.getOrElse(i, 0L)
        val count2Val = count2.getOrElse(i, 0L)
        i -> (count1Val + count2Val)
      }(collection.breakOut) // more efficient alternative to [[collection.Traversable.toMap]]
    }
    def vertexProgram(vid: VertexId, attr: Long, message: Map[VertexId, Long]): VertexId = {
      if (message.isEmpty) attr else message.maxBy(_._2)._1
    }
    val initialMessage = Map[VertexId, Long]()
    Pregel(lpaGraph, initialMessage, maxIterations = maxSteps)(
      vprog = vertexProgram,
      sendMsg = sendMessage,
      mergeMsg = mergeMessage)
  }

  def globalClusteringCoefficient[VD: ClassTag, ED: ClassTag](g:Graph[VD, ED]) = {
    val numerator  = g.triangleCount().vertices.map(_._2).reduce(_ + _)
    val denominator = g.inDegrees.map{ case (_, d) => d*(d-1) / 2.0 }.reduce(_ + _)
    if(denominator == 0) 0.0 else numerator / denominator
  }

  def localClusteringCoefficient[VD: ClassTag, ED: ClassTag](g: Graph[VD, ED]) = {
    val triCountGraph = g.triangleCount()
    val maxTrisGraph = g.inDegrees.mapValues(srcAttr => srcAttr*(srcAttr-1) / 2.0 )
    triCountGraph.vertices.innerJoin(maxTrisGraph){ (vid, a, b) => if(b == 0) 0 else a / b }
  }

}

