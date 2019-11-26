package paristech

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

object Trainer {

  // Train and test, see usage
  def main(args: Array[String]): Unit = {

    object Command extends Enumeration {
      type Command = Value
      val DEFAULT, TRAIN, TEST = Value
    }

    var command = Command.DEFAULT
    var gridSearch = true
    var inputPath = Context.dataPath + "/prepared_trainingset"

    val spark = Context.createSession()

    // Command line args
    var i: Int = 0
    while (i < args.length) {
      args(i) match {
        case "--train" => command = Command.TRAIN

        case "--test" => command = Command.TEST

        case "--single-run" => gridSearch = false

        case "--grid-search" => gridSearch = true

        case "--in" => {
          i += 1
          inputPath = args(i)
        }

        case "target.*\\.jar" => {/* ignore */}

        case _ => {
          print("Unknown argument " + args(i) + "\n")
          print("Usage: --train|--test --single-run|--grid-search\n")
        }
      }
      i += 1
    }

    command match {
      case Command.TRAIN => {
        val Array(dfTrain, dfTest) = splitData(spark, inputPath, saveTestData = true)

        print("\n=== Starting Model fitting ===\n\n")
        fitModel(dfTrain, saveModels = true, gridSearch)
      }
      case Command.TEST => {
        // Read data
        val dfTest: DataFrame = spark.read.parquet(Context.outputPath + "/test/test_df")

        // And load it back in during production
        val model = TrainValidationSplitModel.load(Context.outputPath + "/models/spark-logistic-regression-model")

        print("\n=== Starting Testing ===\n\n")
        test(model, dfTest)
      }
      case _ => {
        // Default : do Train and Test
        val Array(dfTrain, dfTest) = splitData(spark, inputPath, saveTestData = false)

        print("\n=== Starting Model fitting ===\n\n")
        val model = fitModel(dfTrain, saveModels = false, gridSearch)
        print("\n=== Starting Model testing ===\n\n")
        test(model, dfTest)
      }
    }
  }

  // Make pipeline (word TF-IDF, categories, label encoders) + logistic regression
  // Also creating the grid search on the min term frequency (minDF) and the regularization parameter
  def buildPipeline(gridSearch: Boolean): (Pipeline, Array[ParamMap]) = {

    // Handling of the texts :
    // 1. tokenize
    // 2. remove stop words
    // 3. compute TF-IDF
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    val stopWordRemover = new StopWordsRemover()
      .setCaseSensitive(false)
      .setLocale("en")
      .setInputCol("tokens")
      .setOutputCol("cleanedTokens")

    val hashingTF = new HashingTF()
      .setInputCol("cleanedTokens").setOutputCol("textTf").setNumFeatures(0x100000)

    val idf = new IDF().setInputCol("textTf").setOutputCol("tfidf")

    // Handling of categorical columns
    // 1. Encode labels with StringIndexer
    // 2. One-hot encoding
    val countryIndexer = new StringIndexer()
      .setHandleInvalid("keep") // To handle labels that are not in training but are in test
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    val currencyIndexer = new StringIndexer()
      .setHandleInvalid("keep") // To handle labels that are not in training but are in test
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    val oneHotEncoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))

    // Eventually assemble all feature columns into a single column
    //
    val assemblerFinal = new VectorAssembler()
      .setInputCols(Array("days_campaign", "hours_prepa", "goal", "tfidf", "country_onehot", "currency_onehot"))
      .setOutputCol("features")
      .setHandleInvalid("skip")

    // Regression model
    //
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setRegParam(0.0001)
      .setElasticNetParam(0.5)
      .setMaxIter(20)
      // Added
      //.setWeightCol("label_weights")

    // Pipeline assembly
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordRemover, hashingTF, idf,
        countryIndexer, currencyIndexer, oneHotEncoder,
        assemblerFinal,
        lr))

    // Grid search parameters
    val paramGrid = new ParamGridBuilder()
    // this grid will have 3 x 4 = 12 parameter settings for CrossValidator to choose from.
    if (gridSearch) {
      paramGrid
        .addGrid(idf.minDocFreq, Array(55, 75, 95))
        .addGrid(lr.regParam, Array(1e-8, 1e-6, 1e-4, 1e-2))
    }
    else {
      paramGrid
        .addGrid(idf.minDocFreq, Array(0))
    }

    return (pipeline, paramGrid.build())
  }

  // Split cleaned data
  // Return (train, test) data frames
  def splitData(spark: SparkSession, inputPath : String, saveTestData: Boolean): Array[DataFrame] = {

    /*
      import spark.implicits._

      // Read preprocessed data
      val dfPrep: DataFrame = spark.read.parquet(Context.dataPath + "/prepared_trainingset")
      // Added
      val df = dfPrep.withColumn("label_weights", $"final_status" * 2 + 1)
    */
    // Read preprocessed data
    val df: DataFrame = spark.read.parquet(inputPath)

    // Split train / test with random seed
    val Array(dfTrain, dfTest) = df.randomSplit(Array(0.9, 0.1))

    // Save test data
    if (saveTestData) {
      dfTest.write.mode(SaveMode.Overwrite).parquet(Context.outputPath + "/test_df")
    }

    Array(dfTrain, dfTest)
  }

  // Train
  def fitModel(dfTrain: DataFrame, saveModels: Boolean, gridSearch: Boolean): TrainValidationSplitModel = {

    // Make pipeline
    val (pipeline, gridParams) = buildPipeline(gridSearch)

    // Save unfit pipeline to disk
    if (saveModels) {
      pipeline.write.overwrite().save(Context.outputPath + "/model0-unfit")
    }

    printf("  Hyper param search grid size : %d\n", gridParams.length)

    // Evaluation of the model based on F1-score
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      // Could be Changed to "weightedPrecision"
      .setMetricName("f1")

    // Split the train and validation data
    // Will actually rerun the gradient descent with full data
    val validator = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(gridParams)
      // 70% of the data will be used for training and the remaining 30% for validation.
      .setTrainRatio(0.7)
      // Evaluate up to 2 parameter settings in parallel
      .setParallelism(2)

    // Fit the pipeline to training documents.
    val model: TrainValidationSplitModel = validator.fit(dfTrain)

    // Save the fitted pipeline to disk
    if (saveModels) {
      model.write.overwrite().save(Context.outputPath + "/model0-fit")
    }

    return model
  }

  // Perform predictions and compute F1-score
  def test(model: TrainValidationSplitModel, dfTest: DataFrame): Unit = {

    // Compute predictions
    val dfWithSimplePredictions = model.transform(dfTest)

    dfWithSimplePredictions.persist()

    // Compute statistics
    dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()

    // F1-score computation
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    println("F1 score = " + evaluator.evaluate(dfWithSimplePredictions))

    dfWithSimplePredictions.unpersist()
  }
}
