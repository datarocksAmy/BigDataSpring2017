import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
/**
  * =====================================================================
  * CS5542 Big Data Analytics & ApplicationLab
  * Assignment #6 - Image Classification ( Decision Tree )
  * Accuracy and Confusion Matrix for DT Model.
  * #20 Chia-Hui Amy Lin
  * =====================================================================
  */

object ImageClassification_DT {
  def main(args: Array[String]) {
    val IMAGE_CATEGORIES = Array("Sea", "PolarBear", "Fish", "Peacock", "Chameleon")
    System.setProperty("hadoop.home.dir", "C:\\winutils")
    // Turn off Info Logger for Consolexxx
    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);
    val sparkConf = new SparkConf().setAppName("ImageClassification").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)
    val train = sc.textFile("data/train")
    val test = sc.textFile("data/test")
    val parsedData = train.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }
    val testData1 = test.map(line => {
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    })

    // Train Data Section
    val trainingData = parsedData

    val numClasses = 5 // Total categories
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32


    // Decision Tree Model
    val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    // Classify each data points
    val classify1 = testData1.map { line =>
      val prediction = model.predict(line.features)
      (line.label, prediction)
    }
    // Fuzzy Classification : Grouping for each images
    val prediction1 = classify1.groupBy(_._1).map(f => {
      var fuzzy_Pred = Array(0, 0, 0, 0, 0) // have to change to 5 things in the array
      f._2.foreach(ff => {
        fuzzy_Pred(ff._2.toInt) += 1
      })
      var count = 0.0
      fuzzy_Pred.foreach(f => {
        count += f
      })
      var i = -1
      var maxIndex = 5
      val max = fuzzy_Pred.max
      val pp = fuzzy_Pred.map(f => {
        val p = f * 100 / count // Calculate percentage
        i = i + 1
        if (f == max)
          maxIndex = i
        (i, p)
      })
      (f._1, pp, maxIndex) // Using the max to determine the image belongs to this class
    })
    prediction1.foreach(f => {
      println("\n\n\n" + f._1 + " : " + f._2.mkString(";\n"))
    })
    val y: RDD[(Double, Double)] = prediction1.map(f => {
      (f._3.toDouble, f._1)
    })

    y.collect().foreach(println(_))

    val metrics = new MulticlassMetrics(y)


    // Print out Accuracy and generate Confusion Matrix
    println("[ Decision Tree Model ]")
    println("-----------------------------------------------------------")
    println("Accuracy:" + metrics.accuracy) // More data, higher accuracy
    println("Confusion Matrix:")
    println(metrics.confusionMatrix)
    println("-----------------------------------------------------------")

  }
}
