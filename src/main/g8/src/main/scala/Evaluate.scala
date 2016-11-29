package $organization$.$name;format="lower,word"$

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.util.ModelSerializer
import org.slf4j.LoggerFactory
import scopt.OptionParser

case class EvaluateConfig(
  input: String = "",
  modelName: String = ""
)

object EvaluateConfig {
  val parser = new OptionParser[EvaluateConfig]("Evaluate") {
      head("$name;format="lower,word"$ Evaluate", "1.0")

      opt[String]('i', "input")
        .required()
        .valueName("<file>")
        .action( (x, c) => c.copy(input = x) )
        .text("The file with test data.")

      opt[String]('m', "model")
        .required()
        .valueName("<modelName>")
        .action( (x, c) => c.copy(modelName = x) )
        .text("Name of trained model file.")
    }

    def parse(args: Array[String]): Option[EvaluateConfig] = {
      parser.parse(args, EvaluateConfig())
    }
}

object Evaluate {
  private val log = LoggerFactory.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("$name;format="lower,word"$-evaluate")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    EvaluateConfig.parse(args) match {
      case Some(config) =>
        val batchSizePerWorker = 10

        val tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)
          .averagingFrequency(5)
          .workerPrefetchNumBatches(2)            //Async prefetching: 2 examples per worker
          .batchSizePerWorker(batchSizePerWorker)
          .build()

        val model = ModelSerializer.restoreMultiLayerNetwork(config.modelName)
        val sparkNet = new SparkDl4jMultiLayer(sc, model, tm)

        val testData = DataIterators.irisCsv(config.input, sqlContext)

        val eval = sparkNet.evaluate(testData)

        log.info(eval.stats())

      case _ =>
        log.error("Invalid arguments.")
    }
  }
}
