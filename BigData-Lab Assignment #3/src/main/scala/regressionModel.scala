/**
  * ========================================================================================================
  * CS5542 Big Data Analytics & ApplicationLab
  * Assignment #3 - Linear Regression for Machine Learning Task (Chimpanzee Island)
  * Build a linear Model for selected two parameters for chimpanzee's daily movements/activities/interactions.
  * Define your own datasets.
  * #20 Chia-Hui Amy Lin
  * << This is another approach since the first method has a way larger training error and test error is 0 >>
  * << No model output.>>
  * Reference: https://github.com/zapletal-martin/spark-regression
  * ========================================================================================================
  */
package com.test

import com.test.Person
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionModel, LinearRegressionWithSGD}
import org.apache.spark.{SparkConf, SparkContext}


case class Person(rating: String, activityLevel: Double, x_location: Double, y_location: Double)

object SparkPeople extends App {

  System.setProperty("hadoop.home.dir","C:\\winutils");

  def prepareFeatures(people: Seq[Person]): Seq[org.apache.spark.mllib.linalg.Vector] = {
    val maxactivityLevel = people map (_ activityLevel) max
    val maxx_location = people map (_ x_location) max
    val maxy_location = people map (_ y_location) max

    people map (p =>
      Vectors dense(
        //if (p.rating == "A") 0.7 else if (p.rating == "B") 0.7 else 0.3,
        p.activityLevel / maxactivityLevel,
        (p.x_location+p.y_location) / (maxx_location+maxy_location)))
  }

  def prepareFeaturesWithLabels(features: Seq[org.apache.spark.mllib.linalg.Vector]): Seq[LabeledPoint] =
    (0d to 1 by (1d / features.length)) zip (features) map (l => LabeledPoint(l._1, l._2))

  // Define Datasets for 6 Chimpanzees (Chimpanzee Label, Activity Level, Location X-Axis, Location Y-Axis)
  override def main(args: Array[String]): Unit = {
    val people = List(
      Person("A", 0.298, 4.893744, 58.580147),
      Person("B", 0.397, 7.014368, 58.309437),
      Person("C", 0.209, 7.612494, 58.363575),
      Person("D", 0.373, 6.198743, 58.904980),
      Person("E", 0.286,7.014371, 65.239426),
      Person("F", 0.409, 12.195639, 65.239426))

    // Initialize Spark
    val sc = new SparkContext(new SparkConf().setAppName("Chimpanzee linear regression").setMaster("local[*]"))
    val data = sc.parallelize(prepareFeaturesWithLabels(prepareFeatures(people)))

    // Parse Data randomly
    val splits = data randomSplit Array(0.7, 0.3)

    val training = splits(0) cache
    val test = splits(1) cache

    val numTraining = training count
    val numTest = test count

    println(s"Training: $numTraining, test: $numTest.")

    val algorithm = new LinearRegressionWithSGD()
    /*algorithm
      .optimizer
      .setNumIterations(100)
      .setStepSize(1)
      .setUpdater(new SquaredL2Updater())
      .setRegParam(0.1)*/

    val model = algorithm run training

    val prediction = model predict (test map (_ features))

    val predictionAndLabel = prediction zip (test map (_ label))

    println("-------------------------RESULTS-------------------------")

    predictionAndLabel.foreach((result) => println(s"predicted label: ${result._1}, actual label: ${result._2}"))

    data.map(x => s"${x.label},${x.features.toString})").foreach(println)

    val loss = predictionAndLabel.map { case (p, l) =>
      val err = p - l
      err * err
    }.reduce(_ + _)

    val rmse = math.sqrt(loss / numTest)
    println("--------------------------------------------------")
    println(s"Test RMSE = $rmse.")
    println("--------------------------------------------------")
    //model.save(sc, "data\\NewLinearRegressionChimp")
    //val sameModel = LinearRegressionModel.load(sc, "data\\NewLinearRegressionChimp")
    sc.stop()
  }
}