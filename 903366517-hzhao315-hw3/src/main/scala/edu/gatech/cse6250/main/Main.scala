package edu.gatech.cse6250.main

import java.text.SimpleDateFormat

import edu.gatech.cse6250.clustering.Metrics
import edu.gatech.cse6250.features.FeatureConstruction
import edu.gatech.cse6250.helper.{ CSVHelper, SparkHelper }
import edu.gatech.cse6250.model.{ Diagnostic, LabResult, Medication }
import edu.gatech.cse6250.phenotyping.T2dmPhenotype
import org.apache.spark.mllib.clustering.{ GaussianMixture, KMeans, StreamingKMeans }
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{ DenseMatrix, Matrices, Vector, Vectors }
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.io.Source

/**
 * @author Hang Su <hangsu@gatech.edu>,
 * @author Yu Jing <yjing43@gatech.edu>,
 * @author Ming Liu <mliu302@gatech.edu>
 */
object Main {
  def main(args: Array[String]) {
    import org.apache.log4j.{ Level, Logger }

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val spark = SparkHelper.spark
    val sc = spark.sparkContext
    //  val sqlContext = spark.sqlContext

    /** initialize loading of data */
    val (medication, labResult, diagnostic) = loadRddRawData(spark)
    val (candidateMedication, candidateLab, candidateDiagnostic) = loadLocalRawData

    /** conduct phenotyping */
    val phenotypeLabel = T2dmPhenotype.transform(medication, labResult, diagnostic)

    /** feature construction with all features */
    val featureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagnostic),
      FeatureConstruction.constructLabFeatureTuple(labResult),
      FeatureConstruction.constructMedicationFeatureTuple(medication)
    )

    // =========== USED FOR AUTO GRADING CLUSTERING GRADING =============
    // phenotypeLabel.map{ case(a,b) => s"$a\t$b" }.saveAsTextFile("data/phenotypeLabel")
    // featureTuples.map{ case((a,b),c) => s"$a\t$b\t$c" }.saveAsTextFile("data/featureTuples")
    // return
    // ==================================================================

    val rawFeatures = FeatureConstruction.construct(sc, featureTuples)

    val (kMeansPurity, gaussianMixturePurity, streamingPurity) = testClustering(phenotypeLabel, rawFeatures)
    println(f"[All feature] purity of kMeans is: $kMeansPurity%.5f")
    println(f"[All feature] purity of GMM is: $gaussianMixturePurity%.5f")
    println(f"[All feature] purity of StreamingKmeans is: $streamingPurity%.5f")

    /** feature construction with filtered features */
    val filteredFeatureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagnostic, candidateDiagnostic),
      FeatureConstruction.constructLabFeatureTuple(labResult, candidateLab),
      FeatureConstruction.constructMedicationFeatureTuple(medication, candidateMedication)
    )

    val filteredRawFeatures = FeatureConstruction.construct(sc, filteredFeatureTuples)

    val (kMeansPurity2, gaussianMixturePurity2, streamingPurity2) = testClustering(phenotypeLabel, filteredRawFeatures)
    println(f"[Filtered feature] purity of kMeans is: $kMeansPurity2%.5f")
    println(f"[Filtered feature] purity of GMM is: $gaussianMixturePurity2%.5f")
    println(f"[Filtered feature] purity of StreamingKmeans is: $streamingPurity2%.5f")
  }

  def testClustering(phenotypeLabel: RDD[(String, Int)], rawFeatures: RDD[(String, Vector)]): (Double, Double, Double) = {
    import org.apache.spark.mllib.linalg.Matrix
    import org.apache.spark.mllib.linalg.distributed.RowMatrix

    val phenotypeLabel_map = phenotypeLabel.collect.toList.toMap

    println("phenotypeLabel: " + phenotypeLabel.count)
    /** scale features */
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(rawFeatures.map(_._2))
    val features = rawFeatures.map({ case (patientID, featureVector) => (patientID, scaler.transform(Vectors.dense(featureVector.toArray))) })
    println("features: " + features.count)
    val rawFeatureVectors = features.map(_._2).cache()
    println("rawFeatureVectors: " + rawFeatureVectors.count)

    /** reduce dimension */
    val mat: RowMatrix = new RowMatrix(rawFeatureVectors)
    val pc: Matrix = mat.computePrincipalComponents(10) // Principal components are stored in a local dense matrix.
    val featureVectors = mat.multiply(pc).rows

    val densePc = Matrices.dense(pc.numRows, pc.numCols, pc.toArray).asInstanceOf[DenseMatrix]

    def transform(feature: Vector): Vector = {
      val scaled = scaler.transform(Vectors.dense(feature.toArray))
      Vectors.dense(Matrices.dense(1, scaled.size, scaled.toArray).multiply(densePc).toArray)
    }

    /**
     * TODO: K Means Clustering using spark mllib
     * Train a k means model using the variabe featureVectors as input
     * Set maxIterations =20 and seed as 6250L
     * Assign each feature vector to a cluster(predicted Class)
     * Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
     * Find Purity using that RDD as an input to Metrics.purity
     * Remove the placeholder below after your implementation
     */
    //     println("joined input:")
    //     rawFeatures.join(phenotypeLabel).sortByKey().collect.foreach(println)

    val k = 3
    val maxIterations = 20
    val seed = 6250L
    val kmeans = KMeans.train(featureVectors, k, maxIterations, "k-means||", seed)

    val kMeansClusterAssignmentAndLabel = rawFeatures.join(phenotypeLabel).map({ case (patientID, (feature, realClass)) => (kmeans.predict(transform(feature)), realClass) })

    val kMeansPurity = Metrics.purity(kMeansClusterAssignmentAndLabel)

    /**
     * TODO: GMMM Clustering using spark mllib
     * Train a Gaussian Mixture model using the variabe featureVectors as input
     * Set maxIterations =20 and seed as 6250L
     * Assign each feature vector to a cluster(predicted Class)
     * Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
     * Find Purity using that RDD as an input to Metrics.purity
     * Remove the placeholder below after your implementation
     */
    val gmm = new GaussianMixture().setSeed(seed).setMaxIterations(maxIterations).setK(k).run(featureVectors)

    val gaussianMixtureClusterAssignmentAndLabel = rawFeatures.join(phenotypeLabel).map({ case (patientID, (feature, realClass)) => (gmm.predict(transform(feature)), realClass) })

    val gaussianMixturePurity = Metrics.purity(gaussianMixtureClusterAssignmentAndLabel)

    /**
     * TODO: StreamingKMeans Clustering using spark mllib
     * Train a StreamingKMeans model using the variabe featureVectors as input
     * Set the number of cluster K = 3, DecayFactor = 1.0, number of dimensions = 10, weight for each center = 0.5, seed as 6250L
     * In order to feed RDD[Vector] please use latestModel, see more info: https://spark.apache.org/docs/2.2.0/api/scala/index.html#org.apache.spark.mllib.clustering.StreamingKMeans
     * To run your model, set time unit as 'points'
     * Assign each feature vector to a cluster(predicted Class)
     * Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
     * Find Purity using that RDD as an input to Metrics.purity
     * Remove the placeholder below after your implementation
     */

    val decay = 1.0
    val numDimensions = 10
    val centerWeight = 0.5
    val stream = new StreamingKMeans()
      .setK(k).setDecayFactor(decay)
      .setRandomCenters(numDimensions, centerWeight, seed)
      .latestModel().update(featureVectors, decay, "points")

    val streamingClusterAssignmentAndLabel = rawFeatures.join(phenotypeLabel).map({ case (patientID, (feature, realClass)) => (stream.predict(transform(feature)), realClass) })

    val streamKmeansPurity = Metrics.purity(streamingClusterAssignmentAndLabel)

    (kMeansPurity, gaussianMixturePurity, streamKmeansPurity)
  }

  /**
   * load the sets of string for filtering of medication
   * lab result and diagnostics
   *
   * @return
   */
  def loadLocalRawData: (Set[String], Set[String], Set[String]) = {
    val candidateMedication = Source.fromFile("data/med_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    val candidateLab = Source.fromFile("data/lab_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    val candidateDiagnostic = Source.fromFile("data/icd9_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    (candidateMedication, candidateLab, candidateDiagnostic)
  }

  def sqlDateParser(input: String, pattern: String = "yyyy-MM-dd'T'HH:mm:ssX"): java.sql.Date = {
    val dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssX")
    new java.sql.Date(dateFormat.parse(input).getTime)
  }

  def loadRddRawData(spark: SparkSession): (RDD[Medication], RDD[LabResult], RDD[Diagnostic]) = {
    /* the sql queries in spark required to import sparkSession.implicits._ */
    import spark.implicits._
    val sqlContext = spark.sqlContext

    /* a helper function sqlDateParser may useful here */

    /**
     * load data using Spark SQL into three RDDs and return them
     * Hint:
     * You can utilize edu.gatech.cse6250.helper.CSVHelper
     * through your sparkSession.
     *
     * This guide may helps: https://bit.ly/2xnrVnA
     *
     * Notes:Refer to model/models.scala for the shape of Medication, LabResult, Diagnostic data type.
     * Be careful when you deal with String and numbers in String type.
     * Ignore lab results with missing (empty or NaN) values when these are read in.
     * For dates, use Date_Resulted for labResults and Order_Date for medication.
     *
     */

    /**
     * TODO: implement your own code here and remove
     * existing placeholder code below
     */

    /**
     * RDD models
     * case class Diagnostic(patientID: String, date: Date, code: String)
     * case class LabResult(patientID: String, date: Date, testName: String, value: Double)
     * case class Medication(patientID: String, date: Date, medicine: String)
     */
    val currentDirectory = new java.io.File(".").getCanonicalPath
    val encounter = CSVHelper.loadCSVAsTable(spark, "file:///" + currentDirectory + "/data/encounter_INPUT.csv", "encounter")
    val encounter_dx = CSVHelper.loadCSVAsTable(spark, "file:///" + currentDirectory + "/data/encounter_dx_INPUT.csv", "encounter_dx")
    val lab_results = CSVHelper
      .loadCSVAsTable(spark, "file:///" + currentDirectory + "/data/lab_results_INPUT.csv", "lab_results")
    val medication_orders = CSVHelper
      .loadCSVAsTable(spark, "file:///" + currentDirectory + "/data/medication_orders_INPUT.csv", "medication_orders")

    val medication_df = medication_orders.select("Member_ID", "Order_Date", "Drug_Name")
    val medication: RDD[Medication] = medication_df.map(line => Medication(line.getString(0), sqlDateParser(line.getString(1)), line.getString(2).toLowerCase)).rdd

    val lab_results_updated = lab_results.select("Member_ID", "Date_Resulted", "Result_Name", "Numeric_Result")
    val lab_results_no_na = lab_results_updated.na.drop()
    import org.apache.spark.sql.functions._
    val lab_results_df = lab_results_no_na.withColumn("Numeric_Result", regexp_replace(col("Numeric_Result"), ",", ""))
    val labResult: RDD[LabResult] = lab_results_df.map(line => LabResult(line.getString(0), sqlDateParser(line.getString(1)), line.getString(2).toLowerCase, line.getString(3).toDouble)).rdd

    val encounter_joined = encounter
      .join(encounter_dx, lower(encounter.col("Encounter_ID")).equalTo(lower(encounter_dx("Encounter_ID"))))
    val encounter_df = encounter_joined.select("Member_ID", "Encounter_DateTime", "code")
    val diagnostic: RDD[Diagnostic] = encounter_df.map(line => Diagnostic(line.getString(0), sqlDateParser(line.getString(1)), line.getString(2).toLowerCase)).rdd

    (medication, labResult, diagnostic)
  }

}
