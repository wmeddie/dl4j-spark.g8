package $organization$.$name;format="lower,word"$


import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import org.deeplearning4j.spark.util.MLLibUtil
import org.nd4j.linalg.dataset.DataSet

object DataIterators {
  private val irisSchema = StructType(
    Array(
      StructField("c1", DoubleType, nullable = false),
      StructField("c2", DoubleType, nullable = false),
      StructField("c3", DoubleType, nullable = false),
      StructField("c4", DoubleType, nullable = false),
      StructField("label", IntegerType, nullable = false)
    )
  )

  def irisCsv(path: String, sqlContext: SQLContext): JavaRDD[DataSet]  = {
    val csvOptions = Map(
      "header" -> "false",
      "inferSchema" -> "false",
      "delimiter" -> ",",
      "escape" -> null,
      "parserLib" -> "univocity",
      "mode" -> "DROPMALFORMED"
    )

    val irisData = sqlContext.load(
      source = "com.databricks.spark.csv",
      schema = irisSchema,
      options = csvOptions + ("path" -> path)
    ).cache()

    irisData.registerTempTable("iris")

    val irisPoints = sqlContext.sql(
      """
        |SELECT
        | c1,
        | c2,
        | c3,
        | c4,
        | label
        |FROM iris
      """.stripMargin
    ).map(row => {
      LabeledPoint(
        row.getInt(4),
        Vectors.dense(row.getDouble(0), row.getDouble(1), row.getDouble(2), row.getDouble(3))
      )
    }).cache()

    MLLibUtil.fromLabeledPoint(sqlContext.sparkContext, irisPoints, 3)
  }
}
