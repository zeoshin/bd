package edu.gatech.cse6250.phenotyping

import edu.gatech.cse6250.model.{ Diagnostic, LabResult, Medication }
import org.apache.spark.rdd.RDD
/**
 * @author Hang Su <hangsu@gatech.edu>,
 * @author Sungtae An <stan84@gatech.edu>,
 */
object T2dmPhenotype {

  /** Hard code the criteria */
  val T1DM_DX = Set("250.01", "250.03", "250.11", "250.13", "250.21", "250.23", "250.31", "250.33", "250.41", "250.43",
    "250.51", "250.53", "250.61", "250.63", "250.71", "250.73", "250.81", "250.83", "250.91", "250.93")

  val T2DM_DX = Set("250.3", "250.32", "250.2", "250.22", "250.9", "250.92", "250.8", "250.82", "250.7", "250.72", "250.6",
    "250.62", "250.5", "250.52", "250.4", "250.42", "250.00", "250.02")

  val T1DM_MED = Set("lantus", "insulin glargine", "insulin aspart", "insulin detemir", "insulin lente", "insulin nph", "insulin reg", "insulin,ultralente")

  val T2DM_MED = Set("chlorpropamide", "diabinese", "diabanase", "diabinase", "glipizide", "glucotrol", "glucotrol xl",
    "glucatrol ", "glyburide", "micronase", "glynase", "diabetamide", "diabeta", "glimepiride", "amaryl",
    "repaglinide", "prandin", "nateglinide", "metformin", "rosiglitazone", "pioglitazone", "acarbose",
    "miglitol", "sitagliptin", "exenatide", "tolazamide", "acetohexamide", "troglitazone", "tolbutamide",
    "avandia", "actos", "actos", "glipizide")

  /**
   * Transform given data set to a RDD of patients and corresponding phenotype
   *
   * @param medication medication RDD
   * @param labResult  lab result RDD
   * @param diagnostic diagnostic code RDD
   * @return tuple in the format of (patient-ID, label). label = 1 if the patient is case, label = 2 if control, 3 otherwise
   */
  def transform(medication: RDD[Medication], labResult: RDD[LabResult], diagnostic: RDD[Diagnostic]): RDD[(String, Int)] = {
    /**
     * Remove the place holder and implement your code here.
     * Hard code the medication, lab, icd code etc. for phenotypes like example code below.
     * When testing your code, we expect your function to have no side effect,
     * i.e. do NOT read from file or write file
     *
     * You don't need to follow the example placeholder code below exactly, but do have the same return type.
     *
     * Hint: Consider case sensitivity when doing string comparisons.
     */

    val sc = medication.sparkContext

    /** Hard code the criteria */
    // val type1_dm_dx = Set("code1", "250.03")
    // val type1_dm_med = Set("med1", "insulin nph")
    // use the given criteria above like T1DM_DX, T2DM_DX, T1DM_MED, T2DM_MED and hard code DM_RELATED_DX criteria as well

    /** Find CASE Patients */

    val diagnostic_T1 = diagnostic
      .filter(x => T1DM_DX.contains(x.code))
    val diagnostic_T2 = diagnostic
      .filter(x => T2DM_DX.contains(x.code))
    val medication_T1 = medication
      .filter(x => T1DM_MED.contains(x.medicine.toLowerCase))
    val medication_T2 = medication
      .filter(x => T2DM_MED.contains(x.medicine.toLowerCase))
    import org.apache.spark.sql.functions._
    val medication_T1_id_date = medication_T1.map(line => (line.patientID, line.date.toString.replace("-", "").toLong))
    val medication_T2_id_date = medication_T2.map(line => (line.patientID, line.date.toString.replace("-", "").toLong))
    val medication_T1_id_date_first = medication_T1_id_date.reduceByKey(math.min(_, _))
    val medication_T2_id_date_first = medication_T2_id_date.reduceByKey(math.min(_, _))
    val medication_T1_T2_first_joined = medication_T1_id_date_first.join(medication_T2_id_date_first)
    val medication_T2_preceeds_T1 = medication_T1_T2_first_joined.filter(x => x._2._1 > x._2._2)

    val diagnostic_T1_patientID = diagnostic_T1
      .map(line => (line.patientID, 1))
      .map(tup => (tup._1, tup))
      .reduceByKey { case (a, b) => a }
      .map(_._2)

    val diagnostic_T2_patientID = diagnostic_T2
      .map(line => (line.patientID, 1))
      .map(tup => (tup._1, tup))
      .reduceByKey { case (a, b) => a }
      .map(_._2)

    val medication_T1_patientID = medication_T1
      .map(line => (line.patientID, 1))
      .map(tup => (tup._1, tup))
      .reduceByKey { case (a, b) => a }
      .map(_._2)

    val medication_T2_patientID = medication_T2
      .map(line => (line.patientID, 1))
      .map(tup => (tup._1, tup))
      .reduceByKey { case (a, b) => a }
      .map(_._2)

    val diagnostic_T1_patientID_keys = diagnostic_T1_patientID.keys.collect.toSet
    val diagnostic_T2_patientID_keys = diagnostic_T2_patientID.keys.collect.toSet

    val medication_T1_patientID_keys = medication_T1_patientID.keys.collect.toSet
    val medication_T2_patientID_keys = medication_T2_patientID.keys.collect.toSet

    val medication_T2_preceeds_T1_keys = medication_T2_preceeds_T1.keys.collect.toSet

    val casePatients = diagnostic_T2_patientID.filter(x => !diagnostic_T1_patientID_keys.contains(x._1)
      && diagnostic_T2_patientID_keys.contains(x._1)
      && !medication_T1_patientID_keys.contains(x._1)
      || !diagnostic_T1_patientID_keys.contains(x._1)
      && diagnostic_T2_patientID_keys.contains(x._1)
      && medication_T1_patientID_keys.contains(x._1)
      && !medication_T2_patientID_keys.contains(x._1)
      || !diagnostic_T1_patientID_keys.contains(x._1)
      && diagnostic_T2_patientID_keys.contains(x._1)
      && medication_T1_patientID_keys.contains(x._1)
      && medication_T2_patientID_keys.contains(x._1)
      && medication_T2_preceeds_T1_keys.contains(x._1))

    /** Find CONTROL Patients */
    val greater_eq_to_map = Map("hba1c" -> "6.0", "hemoglobin a1c" -> "6.0", "fasting glucose" -> "110",
      "fasting blood glucose" -> "110", "fasting plasma glucose" -> "110")

    val greater_than_map = Map("glucose" -> "110", "glucose, serum" -> "110")

    val DM_RELATED_DX = Set("790.21", "790.22", "790.2", "790.29", "648.81", "648.82", "648.83",
      "648.84", "648", "648.01", "648.02", "648.03", "648.04", "791.5", "277.7",
      "V77.1", "256.4")

    val glucose_measure = labResult
      .filter(x => x.testName.toLowerCase.contains("glucose"))

    val glucose_measure_abnormal1_keys = labResult
      .filter(x => greater_than_map.exists(_._1 == x.testName.toLowerCase)
        && x.value > greater_than_map(x.testName.toLowerCase).toDouble)
      .map(line => (line.patientID, line.value))
      .map(tup => (tup._1, tup))
      .reduceByKey { case (a, b) => a }
      .map(_._2).keys.collect.toSet

    val glucose_measure_abnormal2_keys = labResult
      .filter(x => greater_eq_to_map.exists(_._1 == x.testName.toLowerCase)
        && x.value >= greater_eq_to_map(x.testName.toLowerCase).toDouble)
      .map(line => (line.patientID, line.value))
      .map(tup => (tup._1, tup))
      .reduceByKey { case (a, b) => a }
      .map(_._2).keys.collect.toSet

    val glucose_measure_normal = glucose_measure
      .filter(x => !glucose_measure_abnormal1_keys.contains(x.patientID)
        && !glucose_measure_abnormal2_keys.contains(x.patientID))

    val diabetes_mellis_diagnostic_keys = diagnostic
      .filter(x => DM_RELATED_DX.contains(x.code) || x.code.matches("250\\..*"))
      .map(line => (line.patientID, 2))
      .map(tup => (tup._1, tup))
      .reduceByKey { case (a, b) => a }
      .map(_._2).keys.collect.toSet

    val controlPatients = glucose_measure_normal
      .filter(x => !diabetes_mellis_diagnostic_keys.contains(x.patientID))
      .map(line => (line.patientID, 2))
      .map(tup => (tup._1, tup))
      .reduceByKey { case (a, b) => a }
      .map(_._2)

    val control_keys = controlPatients.keys.collect.toSet
    val case_keys = casePatients.keys.collect.toSet

    /** Find OTHER Patients */
    val others = diagnostic
      .filter(x => !control_keys.contains(x.patientID)
        && !case_keys.contains(x.patientID))
      .map(line => (line.patientID, 3))
      .map(tup => (tup._1, tup))
      .reduceByKey { case (a, b) => a }
      .map(_._2)

    /** Once you find patients for each group, make them as a single RDD[(String, Int)] */
    val phenotypeLabel = sc.union(casePatients, controlPatients, others)

    /** Return */
    phenotypeLabel
  }
}