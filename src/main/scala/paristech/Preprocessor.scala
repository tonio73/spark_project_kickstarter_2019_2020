package paristech

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{udf, lower, when}

object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL
    val spark = Context.createSession()

    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    val df: DataFrame = spark
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv("./cours-spark-telecom/data/train_clean.csv")

    println(s"Nombre de lignes : ${df.count}")
    println(s"Nombre de colonnes : ${df.columns.length}")
    df.printSchema()

    val dfCasted = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline" , $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))

    dfCasted.printSchema()

    dfCasted.groupBy("disable_communication").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("country").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("currency").count.orderBy($"count".desc).show(100)
    dfCasted.select("deadline").dropDuplicates.show()
    dfCasted.groupBy("state_changed_at").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("backers_count").count.orderBy($"count".desc).show(100)
    dfCasted.select("goal", "final_status").show(30)
    dfCasted.groupBy("country", "currency").count.orderBy($"count".desc).show(50)

    val df2 = dfCasted.drop("disable_communication")

    val dfNoFuture = df2.drop("backers_count", "state_changed_at")

    def cleanCountry(country: String, currency: String): String = {
      if(country == "False")
        currency
      else if (country.length > 2)
        null
      else
        country
    }

    def cleanCurrency(currency: String): String = {
      if (currency.length != 3)
        null
      else
        currency
    }

    val cleanCountryUdf = udf(cleanCountry _)
    val cleanCurrencyUdf = udf(cleanCurrency _)

    val dfCountry = dfNoFuture
      .withColumn("country2", cleanCountryUdf($"country", $"currency"))
      .withColumn("currency2", cleanCurrencyUdf($"currency"))
      .drop("country", "currency")

    dfCountry.groupBy("final_status").count.orderBy($"count"desc)show()

    val dfFinalStatus = dfCountry.where($"final_status" === 0 || $"final_status" === 1)

    val dfTimes = dfFinalStatus
      .withColumn("days_campaign", ($"deadline" - $"launched_at") / (3600*24))
      .withColumn("hours_prepa", (($"launched_at" - $"created_at") / 3.6).cast("Int") / 1000.0)
      .drop("launched_at", "created_at", "deadline")

    dfTimes.groupBy("hours_prepa").count.orderBy($"count".desc)show()

    val dfTextNormed = dfTimes
      .withColumn("name", lower($"name"))
      .withColumn("desc", lower($"desc"))
      .withColumn("keywords", lower($"keywords"))
      .withColumn("text", $"name" + " " + $"desc" + " " + $"keywords")

    val dfNoNull = dfTextNormed
      .withColumn("days_campaign", when($"days_campaign".isNull, -1).otherwise($"days_campaign"))
      .withColumn("days_campaign", when($"days_campaign".isNull, -1).otherwise($"days_campaign"))
      .withColumn("hours_prepa", when($"hours_prepa".isNull, -1).otherwise($"hours_prepa"))
      .withColumn("country2", when($"country2".isNull, "unknown").otherwise($"country2"))
      .withColumn("currency2", when($"currency2".isNull, "unknown").otherwise($"currency2"))

    // Eventually write cleaned data to be used in the Trainer
    dfNoNull.write.parquet(Context.dataPath + "/preprocessed")
  }
}
