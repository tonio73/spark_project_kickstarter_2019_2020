# Spark project MS Big Data Télécom : Kickstarter campaigns

Spark project for MS Big Data Telecom based on Kickstarter campaigns 2019-2020

## Build and Run

Input data is now stored within the subdirectory : *data/*

Outputs are written to subdirectory : *output/*

Data location may get changed in scala property paritech.Context.dataPath

### Build 

Either form IntelliJ using the module "mainRunner", or build and run with the SBT script [build.sbt](build.sbt)

### Run

Either from IntelliJ using one of the provided configurations, or through the [build_and_submit.sh](build_and_submit.sh) shell script

Options:
- --train | --test : restrict the processing to training only or testing only. Default : train and test
- --single-run | --grid-search : restrict the train to single run or perform a grid search. Default : grid search
- --in [filepath] specify the path to the parquet file to use as input. Default : data/prepared_trainingset/

## Rapport

### Implementation

- TP2-TP3 initial work performed in Jupyter, the code has been copied over to scala files

- build.sbt modified in order to allow build and launch from IntelliJ, keeping compatibility with the build_and_submit.sh

- command line options added to select options :
    - --single-run|--grid-search to disable/enable grid search (default : grid search enabled)
    - --train|--test to only run the train (and save model and test data to file), or the test from previously saved files (default run train and test in sequence)
    
- Config object 
    - setup Spark session 
    - define the data path (default assuming that the project "cours-spark-telecom" is cloned in the same directory as the current project)
    
### Fitting results
 
#### Without parameter tuning

| F1 score | 0.626 |
| -------- | ----- |
| Precision | 0.42 |
| Recall    | 0.48 |

| final_status | predictions | count |
| ------------ | ----------- | ----- |
|            1 |         0.0 |  1783 |
|            0 |         1.0 |  2347 |
|            1 |         1.0 |  1649 |
|            0 |         0.0 |  5049 |

#### With parameter tuning (grid search)

| F1 score | 0.654 |
| -------- | ----- |
| Precision | 0.46 |
| Recall    | 0.70 |

| final_status | predictions | count |
| ------------ | ----------- | ----- |
|            1 |         0.0 |  1043 | 
|            0 |         1.0 |  2433 | 
|            0 |         0.0 |  4511 |


F1 score is not changing much, however the number of true positive has changed. Actually, recall has jumped from 48% to 70%

### Modifications to the pipeline

#### Preprocessing (function Trainer.buildPipeline())
    
- Add a weight column to overweight the true labels which are less frequent than the false labels
    - transforming values : [0, 1] => [1, 3] (val df = dfPrep.withColumn("label_weights", $"final_status" * 2 + 1))
    - similar to resampling with uneven ratios
- Change optimization metric from "f1" to "weightedPrecision"
    
Computation results with grid search :

F1 score = 0.2822316574658428

| F1 score | 0.282 |
| -------- | ----- |
| Precision | 0.344 |
| Recall    | 0.995 |

|final_status|predictions|count|
|------------ | ----------- | ----- |
|           1|        0.0|   17|
|           0|        1.0| 6539|
|           1|        1.0| 3431|
|           0|        0.0|  685|

The recall is now very good... but the overall precision is awfully bad.

