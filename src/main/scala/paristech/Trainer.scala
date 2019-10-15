package paristech

import org.apache.spark.sql.{DataFrame}
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, StringIndexer,
  OneHotEncoderEstimator, VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
// import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

object Trainer {

  // Make pipeline (word TFIDF, categories, label encoders) + logistic regression
  def buildPipeline(): Pipeline = {

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

    val countryIndexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    val currencyIndexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    val oneHotEncoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))

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
      .setMaxIter(20)

    new Pipeline()
      .setStages(Array(tokenizer, stopWordRemover, hashingTF, idf, countryIndexer,
        currencyIndexer, oneHotEncoder, assembler, lr))
  }

  // Tester : read model as saved,
  def main(args: Array[String]): Unit = {

    val spark = Context.createSession()

    // Read data
    val df: DataFrame = spark.read.parquet(Context.dataPath + "prepared_trainingset")

    // Split train / test
    val Array(dfTrain, dfTest) = df.randomSplit(Array(0.9, 0.1), seed = Context.splitterSeed)

    // Make pipeline
    val pipeline = buildPipeline()

    // Fit the pipeline to training documents.
    val model0 = pipeline.fit(dfTrain)

    // Now we can optionally save the fitted pipeline to disk
    model0.write.overwrite().save(Context.dataPath + "/models/spark-logistic-regression-model")

    // We can also save this unfit pipeline to disk
    pipeline.write.overwrite().save(Context.dataPath + "/models/unfit-lr-model")
  }
}
