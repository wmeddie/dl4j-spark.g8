name := "$name$"

organization := "$organization$"

version := "$version$"

scalaVersion := "2.10.6"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.3.1" % "provided"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.3.1" % "provided"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "1.3.1" % "provided"

libraryDependencies += "com.databricks" %% "spark-csv" % "1.3.0"

// For CPU (Comment out to use the GPU)
libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.7.0"

// For GPU (If you've done the Nvidia cuda dance.)
//libraryDependencies += "org.nd4j" % "nd4j-cuda-8.0-platform" % "0.7.0"
//libraryDependencies += "org.deeplearning4j" % "deeplearning4j-cuda-8.0" % "0.7.0"

libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "0.7.0"

libraryDependencies += "org.deeplearning4j" %% "dl4j-spark" % "0.7.0" intransitive()

libraryDependencies += "com.github.scopt" %% "scopt" % "3.5.0"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % "test"

assemblyMergeStrategy in assembly := {
  case PathList(ps @ _*) if ps.last endsWith ".properties" => MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".xml" => MergeStrategy.first
  case x =>
    val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}