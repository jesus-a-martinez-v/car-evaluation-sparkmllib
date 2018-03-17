import CarEvaluation.spark
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{DataFrame, Dataset, Row}

object Train {
  import spark.sqlContext.implicits._

  def trainAndTest(dataFrame: DataFrame, trainData: Dataset[Row], testData: Dataset[Row]): Unit = {
    val featuresColumns = dataFrame.columns.filterNot(_ == "label")

    val assembler = new VectorAssembler()
      .setInputCols(featuresColumns)
      .setOutputCol("features")

    val classifier = new RandomForestClassifier()

    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.impurity, Array("gini", "entropy"))
      .addGrid(classifier.numTrees, Array(10, 25, 40, 50, 100, 200))
      .addGrid(classifier.minInstancesPerNode, Array(2, 3, 5, 10))
      .build()

    val stages = Array(assembler, classifier)
    val pipeline = new Pipeline().setStages(stages)

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)

    val model = trainValidationSplit.fit(trainData)
    evaluateAndPrintStatistics(model, testData)
  }

  private def evaluateAndPrintStatistics(model: TrainValidationSplitModel, testData: Dataset[Row]): Unit = {
    val results = model.transform(testData)

    val predictionAndLabels = results.select(results("prediction"), results("label")).as[(Double, Double)].rdd

    val metrics = new MulticlassMetrics(predictionAndLabels)

    println("Confusion Matrix: ")
    println(metrics.confusionMatrix)

    println("Accuracy: " + metrics.accuracy)
    println("Weighted False Positive Rate: " + metrics.weightedFalsePositiveRate)
    println("Weighted Precision: " + metrics.weightedPrecision)
    println("Weighted Recall: " + metrics.weightedRecall)
    println("Weighted True Positive Rate: " + metrics.weightedTruePositiveRate)

  }
}
