/**
  * =====================================================================
  * CS5542 Big Data Analytics & ApplicationLab
  * Assignment #3 - K-Mean Clustering
  * Implement K-Mean clustering for the clusters of chimpanzee's activities.
  * User defined dataset. --- chimpanzee_KmeansData.txt (Activity Level, X-Axis Location, Y-Axis Location)
  * #20 Chia-Hui Amy Lin
  * =====================================================================
  */
import org.apache.log4j.{Logger, Level}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

object kMeansClustering {

  def main(args: Array[String]): Unit = {

    System.setProperty("hadoop.home.dir","C:\\winutils");

    // Initialize Spark
    val sparkConf = new SparkConf().setAppName("KMeanClusteringChimpanzee").setMaster("local[*]")
    val sc=new SparkContext(sparkConf)

    // Turn off Info Logger for Consolexxx
    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);

    // Load data sets from chimapanzee_KmreanData.txt" & parse the data
    val data = sc.textFile("data/chimpanzee_KmeansData.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    // Print out training data sets
    parsedData.foreach(f=>println(f))

    // Cluster the data into two classes using K-Means
    val numClusters = 2
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors (WSSSE)
    val WSSSE = clusters.computeCost(parsedData)
    println("------------------------------------------------------------------------------")
    println("Within Set Sum of Squared Errors = " + WSSSE)

    // Make predictions based on training data in the cluster
    println("------------------------------------------------------------------------------")
    println("Clustering on training data: ")
    clusters.predict(parsedData).zip(parsedData).foreach(f=>println(f._2,f._1))
    println("------------------------------------------------------------------------------")

    // Save to file "KMeansModelChimpanzee" and load model
    clusters.save(sc, "data/KMeansModelChimpanzee")
    val sameModel = KMeansModel.load(sc, "data/KMeansModelChimpanzee")


  }


}
