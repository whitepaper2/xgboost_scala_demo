package tongdun

import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.fpm.FPGrowth.FreqItemset
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.reflect.ClassTag
import org.apache.spark.{HashPartitioner, Partitioner, SparkContext, SparkException}

object FPGrowthDemo {

  class FPTree[T] extends Serializable {

    import FPTree._

    val root: Node[T] = new Node(null)

    private val summaries: mutable.Map[T, Summary[T]] = mutable.Map.empty

    /** Adds a transaction with count. */
    def add(t: Iterable[T], count: Long = 1L): this.type = {
      require(count > 0)
      var curr = root
      curr.count += count
      t.foreach { item =>
        val summary = summaries.getOrElseUpdate(item, new Summary)
        summary.count += count
        val child = curr.children.getOrElseUpdate(item, {
          val newNode = new Node(curr)
          newNode.item = item
          summary.nodes += newNode
          newNode
        })
        child.count += count
        curr = child
      }
      this
    }

    /** Merges another FP-Tree. */
    def merge(other: FPTree[T]): this.type = {
      other.transactions.foreach { case (t, c) =>
        add(t, c)
      }
      this
    }

    /** Gets a subtree with the suffix. */
    private def project(suffix: T): FPTree[T] = {
      val tree = new FPTree[T]
      if (summaries.contains(suffix)) {
        val summary = summaries(suffix)
        summary.nodes.foreach { node =>
          var t = List.empty[T]
          var curr = node.parent
          while (!curr.isRoot) {
            t = curr.item :: t
            curr = curr.parent
          }
          tree.add(t, node.count)
        }
      }
      tree
    }

    /** Returns all transactions in an iterator. */
    def transactions: Iterator[(List[T], Long)] = getTransactions(root)

    /** Returns all transactions under this node. */
    private def getTransactions(node: Node[T]): Iterator[(List[T], Long)] = {
      var count = node.count
      node.children.iterator.flatMap { case (item, child) =>
        getTransactions(child).map { case (t, c) =>
          count -= c
          (item :: t, c)
        }
      } ++ {
        if (count > 0) {
          Iterator.single((Nil, count))
        } else {
          Iterator.empty
        }
      }
    }

    /** Extracts all patterns with valid suffix and minimum count. */
    def extract(
                 minCount: Long,
                 validateSuffix: T => Boolean = _ => true): Iterator[(List[T], Long)] = {
      summaries.iterator.flatMap { case (item, summary) =>
        if (validateSuffix(item) && summary.count >= minCount) {
          Iterator.single((item :: Nil, summary.count)) ++
            project(item).extract(minCount).map { case (t, c) =>
              (item :: t, c)
            }
        } else {
          Iterator.empty
        }
      }
    }
  }

  object FPTree {

    /** Representing a node in an FP-Tree. */
    class Node[T](val parent: Node[T]) extends Serializable {
      var item: T = _
      var count: Long = 0L
      val children: mutable.Map[T, Node[T]] = mutable.Map.empty

      def isRoot: Boolean = parent == null
    }

    /** Summary of an item in an FP-Tree. */
    private class Summary[T] extends Serializable {
      var count: Long = 0L
      val nodes: ListBuffer[Node[T]] = ListBuffer.empty
    }

  }

  def genFreqItems[Item: ClassTag](data: RDD[Array[Item]], minCount: Long, partitioner: Partitioner): Array[Item] = {
    data.flatMap { t => t }.map(v => (v, 1L))
      .reduceByKey(partitioner, _ + _)
      .filter(_._2 >= minCount)
      .collect()
      .sortBy(-_._2)
      .map(_._1)
  }

  def genFreqItemsets[Item: ClassTag](data: RDD[Array[Item]], minCount: Long, freqItems: Array[Item],
                                      partitioner: Partitioner): RDD[FreqItemset[Item]] = {
    val itemToRank = freqItems.zipWithIndex.toMap
    data.flatMap { transaction =>
      genCondTransactions(transaction, itemToRank, partitioner)
    }.aggregateByKey(new FPTree[Int], partitioner.numPartitions)(
      (tree, transaction) => tree.add(transaction, 1L),
      (tree1, tree2) => tree1.merge(tree2))
      .flatMap { case (part, tree) =>
        tree.extract(minCount, x => partitioner.getPartition(x) == part)
      }.map { case (ranks, count) =>
      new FreqItemset(ranks.map(i => freqItems(i)).toArray, count)
    }
  }

  def genCondTransactions[Item: ClassTag](transaction: Array[Item], itemToRank: Map[Item, Int],
                                          partitioner: Partitioner): mutable.Map[Int, Array[Int]] = {
    val output = mutable.Map.empty[Int, Array[Int]]
    // Filter the basket by frequent items pattern and sort their ranks.
    val filtered = transaction.flatMap(itemToRank.get)
    java.util.Arrays.sort(filtered)
    val n = filtered.length
    var i = n - 1
    while (i >= 0) {
      val item = filtered(i)
      val part = partitioner.getPartition(item)
      if (!output.contains(part)) {
        output(part) = filtered.slice(0, i + 1)
      }
      i -= 1
    }
    output.foreach(r=>println(r._1, r._2.mkString(",")))
    output
  }

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.master("local[4]").appName("XGBoost4J-Spark Pipeline Example").getOrCreate
    val data = spark.sparkContext.textFile("file:/Users/pengyuan.li/Documents/spark2.3/data/mllib/sample_fpgrowth.txt")

    val transactions: RDD[Array[String]] = data.map(s => s.trim.split(' '))
    val minCount = 2L
    val partitioner = new HashPartitioner(data.partitions.length)
    val freqItems = genFreqItems(transactions, minCount, partitioner)
//        val freqItemsets = genFreqItemsets(transactions, minCount, freqItems, partitioner)
    val itemToRank = freqItems.zipWithIndex.toMap
    val t = transactions.flatMap { transaction =>
      genCondTransactions(transaction, itemToRank, partitioner)
    }.aggregateByKey(new FPTree[Int], partitioner.numPartitions)(
      (tree, transaction) => tree.add(transaction, 1L),
      (tree1, tree2) => tree1.merge(tree2)).flatMap { case (part, tree) =>
      tree.extract(minCount, x => partitioner.getPartition(x) == part)
    }
    println(t.collect())

  }
}
