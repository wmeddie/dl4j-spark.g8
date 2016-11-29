package $organization$.$name;format="lower,word"$

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.SplitTestAndTrain
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import scopt.OptionParser

import java.io.File

case class TrainConfig(
  input: File = null,
  modelName: String = "",
  nEpochs: Int = 1
)

object TrainConfig {
  val parser = new OptionParser[TrainConfig]("Train") {
      head("$name;format="lower,word"$ Train", "1.0")

      opt[File]('i', "input")
        .required()
        .valueName("<file>")
        .action( (x, c) => c.copy(input = x) )
        .text("The file with training data.")

      opt[Int]('e', "epoch")
        .action( (x, c) => c.copy(nEpochs = x) )
        .text("Number of times to go over whole training set.")

      opt[String]('o', "output")
        .required()
        .valueName("<modelName>")
        .action( (x, c) => c.copy(modelName = x) )
        .text("Name of trained model file.")
    }

    def parse(args: Array[String]): Option[TrainConfig] = {
      parser.parse(args, TrainConfig())
    }
}

object Train {
  private val log = LoggerFactory.getLogger(getClass)

  private def net(nIn: Int, nOut: Int) = new NeuralNetConfiguration.Builder()
    .seed(42)
    .iterations(1)
    .activation("relu")
    .weightInit(WeightInit.XAVIER)
    .learningRate(0.1)
    .regularization(true).l2(1e-4)
    .list(
      new DenseLayer.Builder().nIn(nIn).nOut(3).build(),
      new DenseLayer.Builder().nIn(3).nOut(3).build(),
      new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .activation("softmax")
        .nIn(3)
        .nOut(nOut)
        .build()
    )
    .build()
    
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("$name;format="lower,word"$-train")
    val sc = new SparkContext(conf)
    val rootLogger = Logger.getRootLogger
    rootLogger.setLevel(Level.INFO)

    TrainConfig.parse(args) match {
      case Some(config) =>
        log.info("Starting training")

        train(config, sc)

        log.info("Training finished.")
      case _ =>
        log.error("Invalid arguments.")
    }
  }

  private def train(c: TrainConfig, sc: SparkContext): Unit = {
    // TODO
  }
}
