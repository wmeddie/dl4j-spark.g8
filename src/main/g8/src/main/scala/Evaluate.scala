package $organization$.$name;format="lower,word"$

import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.util.ModelSerializer
import org.slf4j.LoggerFactory
import scopt.OptionParser

import java.io.File

case class EvaluateConfig(
  input: File = null,
  modelName: String = ""
)

object EvaluateConfig {
  val parser = new OptionParser[EvaluateConfig]("Evaluate") {
      head("$name;format="lower,word"$ Evaluate", "1.0")

      opt[File]('i', "input")
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
        val model = ModelSerializer.restoreMultiLayerNetwork(config.modelName)
        val testData = DataIterators.irisCsv(config.input, sqlContext)

        val eval = new Evaluation(3)
        while (testData.hasNext) {
            val ds = testData.next()
            val output = model.output(ds.getFeatureMatrix)
            eval.eval(ds.getLabels, output)
        }
        
        log.info(eval.stats())

      case _ =>
        log.error("Invalid arguments.")
    }
  }
}
