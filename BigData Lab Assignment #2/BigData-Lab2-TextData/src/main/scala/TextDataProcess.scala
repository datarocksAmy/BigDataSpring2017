import org.apache.spark.{SparkConf, SparkContext}
/**
  * =====================================================================
  * CS5542 Big Data Analytics & ApplicationLab
  * Assignment #2 - Text Data
  * Use at least 2 Spark Transformation & 2 Spark Action
  * #20 Chia-Hui Amy Lin
  * =====================================================================
  */

object TextDataProcess {
  def main(args: Array[String]): Unit = {

    System.setProperty("hadoop.home.dir", "C:\\winutils");

    // Initializing Spark
    val sparkConf = new SparkConf().setAppName("TextDataProcess").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)

    // Read in Text Data from textdata.txt
    val textdata = sc.textFile("textdata.txt")

    // Splitting and Mapping Words
    val wc = textdata.flatMap(line => {line.split(" ")}).map(word => (word, 1)).cache()

    // Reducing: Add up words count with the same Word
    val output = wc.reduceByKey(_ + _)

    // Output result to "output.txt" as Text File format
    output.repartition(1).saveAsTextFile("output.txt")

    // Collect all results as an array
    val result = output.collect()

    // Print out in result Console
    var s: String = "----------------\n Words : Count \n---------------- \n"
    result.foreach { case (word, count) => { s += word + " : " + count + "\n" }}

    println(s)


  }
}