/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse6250.clustering

import org.apache.spark.rdd.RDD

object Metrics {
  /**
   * Given input RDD with tuples of assigned cluster id by clustering,
   * and corresponding real class. Calculate the purity of clustering.
   * Purity is defined as
   * \fract{1}{N}\sum_K max_j |w_k \cap c_j|
   * where N is the number of samples, K is number of clusters and j
   * is index of class. w_k denotes the set of samples in k-th cluster
   * and c_j denotes set of samples of class j.
   *
   * @param clusterAssignmentAndLabel RDD in the tuple format
   *                                  (assigned_cluster_id, class)
   * @return
   */
  def purity(clusterAssignmentAndLabel: RDD[(Int, Int)]): Double = {
    /**
     * TODO: Remove the placeholder and implement your code here
     */
    val N = clusterAssignmentAndLabel.count.toDouble

    val clusterAssignmentAndLabel_group = clusterAssignmentAndLabel
      .map(x => ((x._1, x._2), 1.0))
      .groupByKey()
      .map(t => (t._1, t._2.sum))
      .map { case ((cluster, cls), sum) => (cluster, (cls, sum)) }
      .reduceByKey { case (a, b) => if (a._2 > b._2) a else b }

    val purity = clusterAssignmentAndLabel_group
      .map { case (k, (c, n)) => (1, n) }
      .groupByKey()
      .map(t => (t._1, t._2.sum))
      .first._2.toDouble / N

    purity
  }
}
