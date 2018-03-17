name := "car-evaluation-spark-mllib"

version := "0.1"

scalaVersion := "2.11.11"

libraryDependencies ++= {
  val sparkVersion = "2.3.0"

  Seq(
    "org.apache.spark" %% "spark-core" % sparkVersion,
    "org.apache.spark" % "spark-sql_2.11" % sparkVersion,
    "org.apache.spark" %% "spark-mllib" % sparkVersion % "compile"
  )
}