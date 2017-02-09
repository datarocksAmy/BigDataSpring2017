/**
  * =====================================================================
  * CS5542 Big Data Analytics & ApplicationLab
  * Assignment #3 - Linear Regression for Machine Learning Task (Chimpanzee Island)
  * Build a linear Model for selected two parameters for chimpanzee's daily movements/activities/interactions.
  * Define your own datasets.
  * #20 Chia-Hui Amy Lin
  * =====================================================================
  */

// Import Libraries
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD


object LinearRegressionwithSGD {

  def main(args: Array[String]): Unit ={

    System.setProperty("hadoop.home.dir","C:\\winutils");

    // Initializing Spark
    val sparkConf = new SparkConf().setAppName("LinearModel").setMaster("local[*]")
    val sc=new SparkContext(sparkConf)

    // Turn off Info Logger for Consolexxx
    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);

    // Load and parse the Chimpanzee_data --> I set this dataset for only 6 chimpanzees (Chimpanzee Label, Activity Level, Location X-Axis, Location Y-Axis)
    val data = sc.textFile("data\\Chimpanzee_data.data")
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }.cache()

    parsedData.take(1).foreach(f=>println(f))

    // Split data into training (95%) and test (5%).
    val Array(training, test) = parsedData.randomSplit(Array(0.7, 0.3))

    // Building the model
    val numIterations = 100
    val stepSize = 0.00000001
    val model = LinearRegressionWithSGD.train(training, numIterations, stepSize)

    // Evaluate model on training examples and compute training error
    val valuesAndPreds = training.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val MSE = valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2) }.mean()
    println("training Mean Squared Error = " + MSE)

    // Evaluate model on test examples and compute training error
    val valuesAndPreds2 = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val MSE2 = valuesAndPreds2.map{ case(v, p) => math.pow((v - p), 2) }.mean()
    println("test Mean Squared Error = " + MSE2)

    // Save model as "LinearRegressionChimpanzees" & load model
    model.save(sc, "data\\LinearRegressionChimpanzees")
    val sameModel = LinearRegressionModel.load(sc, "data\\LinearRegressionChimpanzees")
  }

}
