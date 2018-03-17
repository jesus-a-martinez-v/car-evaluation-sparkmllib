import java.io.File

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession

object Test extends App {
  implicit val spark: SparkSession = getSparkSession
  import spark.sqlContext.implicits._

  val data = getData(new File(this.getClass.getClassLoader.getResource("car.data").toURI).getPath)
  data.printSchema()

  val numberOfRows = data.count()
  val numberOfColumns = data.columns.length
  val numberOfFeatures = numberOfColumns - 1

  println(s"Number of rows: $numberOfRows")
  println(s"Number of columns: $numberOfColumns")
  println(s"Number of features: $numberOfFeatures")

  val labelIndexer = new StringIndexer().setInputCol("eval").setOutputCol("label")
  val dataWithLabel = labelIndexer.fit(data).transform(data)

  dataWithLabel.printSchema()

  val featuresColumns = dataWithLabel.columns.filterNot(_ == "label")

  val Array(trainingData, testData) = dataWithLabel.randomSplit(Array(0.75, 0.25), seed = 42)

  val assembler = new VectorAssembler()
    .setInputCols(featuresColumns.map(_ + "Vec"))
    .setOutputCol("features")

  val classifier = new RandomForestClassifier()

  val paramGrid = new ParamGridBuilder()
    .addGrid(classifier.impurity, Array("gini", "entropy"))
    .addGrid(classifier.numTrees, Array(10, 25, 40, 50, 100, 200))
    .addGrid(classifier.minInstancesPerNode, Array(2, 3, 5, 10))
    .build()

  val stages = (
    featuresColumns.map(getStringIndexer).toList :::
    featuresColumns.map(getOneHotEncoder).toList :::
      List(assembler, classifier)).toArray
  val pipeline = new Pipeline().setStages(stages)

  val trainValidationSplit = new TrainValidationSplit()
    .setEstimator(pipeline)
    .setEvaluator(new MulticlassClassificationEvaluator())
    .setEstimatorParamMaps(paramGrid)
    .setTrainRatio(0.8)

  val model = trainValidationSplit.fit(trainingData)
  val results = model.transform(testData)

  val predictionAndLabels = results.select(results("prediction"), results("label")).as[(Double, Double)].rdd
  val metrics = new MulticlassMetrics(predictionAndLabels)
  val fScore = metrics.fMeasure(1.0)

  println(fScore)
  println("Confusion Matrix: ")
  println(metrics.confusionMatrix)
  println("accuracy " + metrics.accuracy)
  println("weightedFalsePositiveRate " + metrics.weightedFalsePositiveRate)

  spark.stop()

  def getSparkSession = SparkSession
    .builder()
    .master("local[*]")
    .appName("CarEvaluation")
    .getOrCreate()

  def getData(path: String) =
    spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("csv")
      .load(path)

  def getStringIndexer(columnName: String) = new StringIndexer()
    .setInputCol(columnName)
    .setOutputCol(s"${columnName}Index")

  def getOneHotEncoder(columnName: String) = new OneHotEncoder()
    .setInputCol(s"${columnName}Index")
    .setOutputCol(s"${columnName}Vec")
}