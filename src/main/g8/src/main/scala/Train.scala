package $organization$.$name;format="lower,word"$

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory
import scopt.OptionParser

case class TrainConfig(
  input: String = null,
  modelName: String = "",
  nEpochs: Int = 1
)

object TrainConfig {
  val parser = new OptionParser[TrainConfig]("Train") {
      head("$name;format="lower,word"$ Train", "1.0")

      opt[String]('i', "input")
        .required()
        .valueName("<file>")
        .action( (x, c) => c.copy(input = x) )
        .text("The file/hdfs path with training data.")

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
    val sqlContext = new SQLContext(sc)
    val train = DataIterators.irisCsv(c.input, sqlContext)

    val batchSizePerWorker = 10

    val tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)
            .averagingFrequency(5)
            .workerPrefetchNumBatches(2)            //Async prefetching: 2 examples per worker
            .batchSizePerWorker(batchSizePerWorker)
            .build()

    val conf = net(4, 3)
    val sparkNet = new SparkDl4jMultiLayer(sc, conf, tm)
    
    for (i <- 0 until c.nEpochs) {
      sparkNet.fit(train.toJavaRDD())
      println(s"Finished epoch $"$"$i.")
    }

    tm.deleteTempFiles(sc)

    ModelSerializer.writeModel(sparkNet.getNetwork, c.modelName, true)

    log.info(s"Model saved to: $"$"${c.modelName}")
  }
}
