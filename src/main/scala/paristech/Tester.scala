package paristech

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.DataFrame

object Tester {

  def main(args: Array[String]): Unit = {

    val spark = Context.createSession()

    // Read data
    val dfTest: DataFrame = spark.read.parquet(Context.dataPath + "/test/test_df")

    // And load it back in during production
    val savedModel = PipelineModel.load(Context.dataPath + "/models/spark-logistic-regression-model")

    val dfWithSimplePredictions = savedModel.transform(dfTest)

    dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    println( "F1 score = " + evaluator.evaluate(dfWithSimplePredictions))
  }
}
