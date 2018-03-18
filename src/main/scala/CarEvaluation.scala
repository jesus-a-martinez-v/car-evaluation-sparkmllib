import org.apache.spark.sql.SparkSession

object CarEvaluation extends App {
  withSparkSession { session =>
    implicit val sparkSession: SparkSession = session

    val data = Data.loadCarData()
    Data.printStatisticsAndSchema(data)

    val preprocessedData = Data.convertOrdinalToIndices(data)
    val dataWithLabel = preprocessedData.withColumnRenamed("eval", "label")
    dataWithLabel.printSchema()

    val (trainingData, testData) = Data.trainTestSplit(dataWithLabel)

    Train.trainAndTest(dataWithLabel, trainingData, testData)
  }

  private def getSparkSession = {
    val sparkSession = SparkSession
      .builder()
      .master("local[*]")
      .appName("CarEvaluation")
      .getOrCreate()

    sparkSession.sparkContext.setLogLevel("ERROR")
    sparkSession
  }

  private def withSparkSession(f: SparkSession => Unit): Unit = {
    implicit val spark: SparkSession = getSparkSession
    f(spark)
    spark.stop()
  }
}