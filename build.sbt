ThisBuild / organization := "paristech"
ThisBuild / scalaVersion := "2.11.11"

val sparkVersion = "2.4.4"
val HadoopVersion = "2.7.2"

// Root project
lazy val root = (project in file("."))
  .settings(
    name := "spark_project_kickstarter_2019_2020",
    version := "1.0"
  )

// Raw spark dependencies to be used either as "provided" or "compiled"
lazy val sparkDependencies = Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion
)

// Other common dependencies
libraryDependencies ++= Seq(

  // Third-party libraries
  "org.apache.hadoop" % "hadoop-aws" % "2.6.0" % "provided",
  "com.amazonaws" % "aws-java-sdk" % "1.7.4" % "provided"
  //"com.github.scopt" %% "scopt" % "3.4.0"        // to parse options given to the jar in the spark-submit
)

// When building assembly to be submitted to Spark, do not include the Spark libs in the Jar
libraryDependencies ++= sparkDependencies.map(_ % "provided")

// Configuration to be used in debug from IntelliJ, must include the Spark Jars
// Must select this in the run configuration on IntelliJ as parameter "Use classpath of module: mainRunner"
// https://github.com/JetBrains/intellij-scala/wiki/%5BSBT%5D-How-to-use-provided-libraries-in-run-configurations
lazy val mainRunner = project.in(file("mainRunner")).dependsOn(RootProject(file("."))).settings(
  libraryDependencies ++= sparkDependencies.map(_ % "compile")
)

// A special option to exclude Scala itself form our assembly JAR, since Spark-submit or IntelliJ already bundle Scala.
assembly / assemblyOption := (assembly / assemblyOption).value.copy(includeScala = false)

// Disable parallel execution because of spark-testing-base
Test / parallelExecution := false

// Configure the build to publish the assembly JAR
(Compile / assembly / artifact) := {
  val art = (Compile / assembly / artifact).value
  art.withClassifier(Some("assembly"))
}

addArtifact(Compile / assembly / artifact, assembly)
