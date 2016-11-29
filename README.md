A [giter8](https://github.com/n8han/giter8) template for creating a scala-based [DeepLearning4J](https://github.com/deeplearning4j/deeplearning4j) [Spark](https://github.com/apache/spark) project.

You use this template by first installing sbt and running  `sbt new wmeddie/dl4j-spark.g8`.

This template includes:

* sbt 0.13.13
* [Spark](https://github.com/apache/spark) Spark 1.3.1 (Should work for up to 1.6)
* [DataVec](https://github.com/deeplearning4j/DataVec) For data ETL and vectorization.
* [DeepLearning4J](https://github.com/deeplearning4j/deeplearning4j) for the actual Deep Learning.
* [ND4J](https://github.com/deeplearning4j/nd4j) the SMID/GPU accelerated DL4J backend library.
* [ScalaTest](http://www.scalatest.org/) For testing.
* Apache 2.0 LICENSE
* A README.md showing how to reproduce the model.

This repo is a personalized version of the existing solutions. 
Inspired from both [chrislewis/basic-project.g8](https://github.com/chrislewis/basic-project.g8) and [softprops/unfiltered.g8](https://github.com/softprops/unfiltered.g8)
