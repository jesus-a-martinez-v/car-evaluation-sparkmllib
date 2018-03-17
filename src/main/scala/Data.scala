import java.io.File

import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object Data {

  def loadCarData()(implicit spark: SparkSession): DataFrame = getData(new File(this.getClass.getClassLoader.getResource("car.data").toURI).getPath)

  def printStatisticsAndSchema(dataFrame: DataFrame): Unit = {
    dataFrame.printSchema()

    val numberOfRows = dataFrame.count()
    val numberOfColumns = dataFrame.columns.length
    val numberOfFeatures = numberOfColumns - 1

    println(s"Number of rows: $numberOfRows")
    println(s"Number of columns: $numberOfColumns")
    println(s"Number of features: $numberOfFeatures")
  }

  def convertOrdinalToIndices(dataFrame: DataFrame): DataFrame =
    dataFrame.columns.foldLeft(dataFrame) { (df, columnName) =>
      val indexerFunction = getColumnIndexer(dataFrame, columnName)
      val column = df(columnName)

      df.withColumn(columnName, indexerFunction.apply(column))
    }

  def trainTestSplit(dataFrame: DataFrame, trainProportion: Double = 0.75, randomSeed: Int = 42): (Dataset[Row], Dataset[Row]) = {
    require(0 < trainProportion && trainProportion < 1)

    val Array(trainingData, testData) = dataFrame.randomSplit(Array(trainProportion, 1 - trainProportion), seed = randomSeed)

    (trainingData, testData)
  }

  private def getData(path: String)(implicit spark: SparkSession) =
    spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("csv")
      .load(path)

  private def getColumnIndexer(dataFrame: DataFrame, columnName: String) = {
    val values: List[String] = getValuesForColumn(columnName)
    val mapping = getOrdinalMapping(values)

    println(s"Mapping for column $columnName")
    println(s"$mapping\n")

    udf((value: String) => mapping(value))
  }

  private def getValuesForColumn(column: String) = column.toLowerCase() match {
    case "buying" | "maint" => List("low", "med", "high", "vhigh")
    case "doors" => List("2", "3", "4", "5more")
    case "persons" => List("2", "4", "more")
    case "lug_boot" => List("small", "med", "big")
    case "safety" => List("low", "med", "high")
    case "eval" => List("unacc", "acc", "good", "vgood")
  }

  private def getOrdinalMapping(values: List[String]) = Map[String, Int](values.zipWithIndex:_*)
}
