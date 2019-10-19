package paristech

import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.feature.{HashingTF, IDF, OneHotEncoderEstimator, RegexTokenizer, StopWordsRemover, StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
// import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

object Trainer {

  // Make pipeline (word TFIDF, categories, label encoders) + logistic regression
  def buildPipeline(): Pipeline = {

    // Handling of the texts :
    // 1. tokenization
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
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

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
      .setRegParam(0.001)
      .setMaxIter(30)

    new Pipeline()
      .setStages(Array(tokenizer, stopWordRemover, hashingTF, idf, countryIndexer,
        currencyIndexer, oneHotEncoder, assembler, lr))
  }

  // Split cleaned data
  // Return (train, test) data frames
  def splitData(spark: SparkSession, saveTestData: Boolean): Array[DataFrame] = {
    // Read preprocessed data
    val df: DataFrame = spark.read.parquet(Context.dataPath + "/prepared_trainingset")

    // Split train / test with random seed
    val Array(dfTrain, dfTest) = df.randomSplit(Array(0.9, 0.1))

    // Save test data
    if (saveTestData) {
      dfTest.write.mode(SaveMode.Overwrite).parquet(Context.dataPath + "/test_df")
    }

    Array(dfTrain, dfTest)
  }

  // Train
  def fitModel(dfTrain: DataFrame, saveModels: Boolean): PipelineModel = {
    // Make pipeline
    val pipeline = buildPipeline()

    // Save unfit pipeline to disk
    if (saveModels) {
      pipeline.write.overwrite().save(Context.dataPath + "/model0-unfit")
    }

    // Fit the pipeline to training documents.
    val model = pipeline.fit(dfTrain)

    // Save the fitted pipeline to disk
    if (saveModels) {
      model.write.overwrite().save(Context.dataPath + "/model0-fit")
    }

    return model
  }

  // Perform predictions and compute F1-score
  def test(model: PipelineModel, dfTest: DataFrame): Unit = {

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

  // Tester : read model as saved,
  def main(args: Array[String]): Unit = {

    val spark = Context.createSession()

    if (args.length > 0) {
      args(0) match {
        case "--train" => {
          val Array(dfTrain, dfTest) = splitData(spark, true)
          fitModel(dfTrain, true)
        }
        case "--test" => {
          // Read data
          val dfTest: DataFrame = spark.read.parquet(Context.dataPath + "/test/test_df")

          // And load it back in during production
          val model = PipelineModel.load(Context.dataPath + "/models/spark-logistic-regression-model")

          test(model, dfTest)
        }
        case _ => {
          // Default : do Train and Test
          val Array(dfTrain, dfTest) = splitData(spark, false)

          print("\n=== Starting Model fitting ===\n\n")
          val model = fitModel(dfTrain, false)
          print("\n=== Starting Model testing ===\n\n")
          test(model, dfTest)
        }
      }
    }
    else {
      // Default : do Train and Test
      val Array(dfTrain, dfTest) = splitData(spark, false)

      print("\n=== Starting Model fitting ===\n\n")
      val model = fitModel(dfTrain, false)
      print("\n=== Starting Model testing ===\n\n")
      test(model, dfTest)
    }
  }
}
