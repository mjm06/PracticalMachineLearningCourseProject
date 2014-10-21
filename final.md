###Practical Machine Learning: Course Project

###Executive Summary

The goal of this project is build a machine learning algorithm using a testing set consisting of measurements of arm, forearm, body, and dumbbell 3-D motion (pitch, yaw, and roll) and acceleration during repetitions of a bicep curl exercise performed in five different ways (A-E), by six experimental subjects (with each subject repeating each exercise type ten times). 

The original training dataset was further subdivided into a training and validation dataset, for the purpose of cross validation and to permit the calculation of in-sample error on the validation set and an estimation of out-of-sample error.

Finally, this algorithm was then used to predict whether twenty exercises in a separate testing set should be classified as A-E.

All steps of this process, including loading, cleaning, and exploratory analysis of the data, along with the model building process on the training set and the model accuracy assessment on the validation set, are described in the sections that follow.



###Loading the Data

The training set and the testing set were loaded from the course website using the following code:

```r
## Download the training set.
if (!file.exists("data")) {
      dir.create("data")
}
fileUrl<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv?accessType=DOWNLOAD"
download.file(fileUrl, destfile="./data/train.csv", method="curl")
dateDownloaded<-date()

## Download the testing set.
if (!file.exists("data")) {
      dir.create("data")
}
fileUrl<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv?accessType=DOWNLOAD"
download.file(fileUrl, destfile="./data/test.csv", method="curl")
dateDownloaded<-date()
```

Then, the training data and the testing data were read into the R environment using the ```read.table``` function, to create the ```trainData``` and ```testData``` dataframes, respectively. 

```r
trainData<-read.table("./data/train.csv", sep=",", header=TRUE)
testData<-read.table("./data/test.csv", sep=",", header=TRUE)
```

###Brief Description of the Data###

```trainData``` and ```testData``` consist of kinematic measurements made when research subjects performed a bicep curl exercise. However, ```testData``` was set aside and not examined in any part of the following pre-processing steps.

To collect the data, the research subjects wore motion capture sensors on their forearms, upper arms, and around their waists; the barbell was also wired with a motion capture sensor. Because each sensor contained a gyroscope, the measurements for each sensor were expressed as three-dimensional motion; specifically, roll, pitch, and yaw. In addition, summary measurements were calculated, such as average, standard deviation, kurtosis, and skewness. The result is that ```trainData``` and ```testData``` contain a total of 159 potential predictor variables (the outcome variable in ```trainData``` is the ```classe``` variable). 

The A-E values of the ```classe``` outcome variable translate to the different ways the bicep curl exercise was performed:
* "A" performed the exercise totally correctly (implying that the hips are motionless; the upper arm is held in a fixed position with the humerus parallel to the trunk of the body and perpendicular to the floor throughout the entire exercise);
* "B" throwing the elbows to the front;
* "C" lifting the dumbbell only halfway;
* "D" lowering the dumbbell only halfway;
* "E" throwing the hips to the front.

###Pre-Processing the Data###
Before the data were pre-processed to faciliate building a model, the initial training set ```trainData``` was split into ```newTrain``` and ```newTest``` datasets.


```r
set.seed(444)
inTrain<-createDataPartition(y=trainData$classe, p=0.7, list=FALSE)

newTrain<-trainData[inTrain, ]
newTest<-trainData[-inTrain, ]
```


All pre-processing decisions were made after examining ```newTrain``` only (although all pre-processing steps were applied to both ```newTrain``` and ```newTest```.) Exploratory analysis and model building were conducted on ```newTrain``` only. ```newTest``` was then used as a validation set to evaluate the model.

Examining the data in ```newTrain``` revealed many variables with values of NA for the majority of rows in the dataset. This is problematic because these NAs will cause some model building methods to fail. Therefore, NAs were replaced with zeroes to maximize the number of variables that can be used to create the machine learning algorithm. Then, all predictor variables were set to numeric class.


```r
classe<-newTrain$classe
predictor.varsTrain<-sapply(newTrain, is.numeric)
newTrain<-cbind(newTrain[,predictor.varsTrain], classe)
newTrain[is.na(newTrain)]<-0
```

The same pre-processing steps were done for ```newTest```, although these data were not viewed.

```r
classe<-newTest$classe
predictor.varsTest<-sapply(newTest, is.numeric)
newTest<-cbind(newTest[,predictor.varsTest], classe)
newTest[is.na(newTest)]<-0
```



###Exploratory Data Analysis
```newTrain``` was then evaluated by making a series of exploratory plots to identify a set of variables that allow the A-E values of ```classe``` to be separated from one another. First, a pairs plot was made, using the simplest and most intuitive of the kinematic measurements:  ```roll_dumbbell```, ```pitch_dumbbell```, ```yaw_dumbbell```, ```roll_arm```, ```pitch_arm```, ```yaw_arm```, ```total_accel_belt```, ```roll_forearm```, ```pitch_forearm```, and ```yaw_forearm```:


```r
cols<-c("red", "green", "blue", "orange", "yellow")
pairs(newTrain[, c("roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", "roll_arm", 
                   "pitch_arm", "yaw_arm", "total_accel_belt", "roll_forearm",
                   "pitch_forearm", "yaw_forearm")], col=cols[newTrain$classe], pch=19)
```

![plot of chunk unnamed-chunk-7](figure/unnamed-chunk-7.png) 


```r
with(data=newTrain, plot(roll_dumbbell, pitch_forearm, col=cols[newTrain$classe], pch=19))
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-81.png) 

```r
with(data=newTrain, plot(yaw_dumbbell, pitch_forearm, col=cols[newTrain$classe], pch=19))
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-82.png) 

```r
with(data=newTrain, plot(roll_dumbbell, roll_forearm, col=cols[newTrain$classe], pch=19))
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-83.png) 


Before building the model, I also determined which combinations of variables seemed the most sensitive due to differences among the research subjects. For example, the following plots show that the variable ```amplitude_pitch_belt``` clearly splits the ```classe``` values E and D according to user name, suggesting that this variable should *not* be included in the predictive model (even though ```amplitude_pitch_belt``` seems to do a great job of separating Class E from the other exercise types -- which makes sense, as exercise E involves moving the trunk of the body while exercises A-D do not). 

```r
qplot(total_accel_belt,  amplitude_pitch_belt, colour=classe, 
      shape=user_name, data=miniTrain, size=I(5))
```

```
## Error: object 'miniTrain' not found
```

```r
#Class E and D are in two distinctly separated clouds according to user_name.

qplot( amplitude_pitch_belt, avg_roll_dumbbell, colour=classe, 
       shape=user_name, data=miniTrain, size=I(5))
```

```
## Error: object 'miniTrain' not found
```

While principal components analysis (PCA) could reduce the number of variables in the dataset, I chose to not conduct PCA so that the final model would be built using variables that are intuitive. The aim of collecting these data, after all, is to quantify exercise performance; therefore, I suggest that the most applicable predictive model for these data will rely on predictors that have intuitive physical meaning (yaw, pitch, roll, and straight-line acceleration), even if some model accuracy is sacrificed.


###Building the Predictive Model

Several models were fit to the training set, using total_accel_belt, roll_dumbbell, pitch_dumbbell, yaw_dumbbell,roll_arm,pitch_arm,yaw_arm,roll_forearm, pitch_forearm and yaw_forearm as the predictor variables. Several algorithm types, including bagging, randomw forests, and boosting, and different cross validation methods (including leave one out cross validation and repeated cross validation) were run. The best model, which exhibited greater than 95% accuracy for all validation classes (as shown below), was created using a random forest approach, with an  out of bag (oob) error estimate included. 


```r
train_control7<-trainControl(method="oob", 
                             classProbs=TRUE, 
                             summaryFunction=twoClassSummary)

modFit7<-train(classe~total_accel_belt+
                     roll_dumbbell+
                     pitch_dumbbell+
                     yaw_dumbbell+
                     roll_arm+
                     pitch_arm+
                     yaw_arm+
                     roll_forearm+
                     pitch_forearm+
                     yaw_forearm,
               data=newTrain,
               method="rf",
               trControl=train_control7,
               tuneLength=15,
               metric="kappa",
               keep.forest=TRUE)
```

```
## note: only 9 unique complexity parameters in default grid. Truncating the grid to 9 .
```

```
## Warning: The metric "kappa" was not in the result set. Accuracy will be
## used instead.
```


Plotting the model's accuracy against the number of trees produced by the random forests method illustrates that the model is over 95% accurate:

![plot of chunk unnamed-chunk-11](figure/unnamed-chunk-11.png) 

Then, the best model ```modFit7``` is applied to the validation set ```newTest```, and the model's performance on the validation set is evaluated.


A confusion matrix for the predicted and observed values of ```classe``` in ```newTest``` illustrates the model's accuracy of at least 95% for each value of ```classe```, as shown below:

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1617   36    7    3    3
##          B   28 1053   21    8   11
##          C   16   28  960   35   11
##          D    7    8   31  902   24
##          E    6   14    7   16 1033
## 
## Overall Statistics
##                                        
##                Accuracy : 0.946        
##                  95% CI : (0.94, 0.951)
##     No Information Rate : 0.284        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.931        
##  Mcnemar's Test P-Value : 0.341        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.966    0.924    0.936    0.936    0.955
## Specificity             0.988    0.986    0.981    0.986    0.991
## Pos Pred Value          0.971    0.939    0.914    0.928    0.960
## Neg Pred Value          0.986    0.982    0.986    0.987    0.990
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.275    0.179    0.163    0.153    0.176
## Detection Prevalence    0.283    0.190    0.178    0.165    0.183
## Balanced Accuracy       0.977    0.955    0.959    0.961    0.973
```

```
##          obs.ans7
## pred.ans7    A    B    C    D    E
##         A 1617   36    7    3    3
##         B   28 1053   21    8   11
##         C   16   28  960   35   11
##         D    7    8   31  902   24
##         E    6   14    7   16 1033
```


The ```varImp``` function reveals the variables of the predictive model that have the greatest impact on the outcome variable ```classe```:

```
## rf variable importance
## 
##                  Overall
## pitch_forearm     100.00
## roll_forearm       77.43
## roll_dumbbell      48.29
## total_accel_belt   45.17
## roll_arm           38.74
## yaw_dumbbell       33.69
## yaw_arm            24.29
## pitch_dumbbell     11.41
## yaw_forearm         5.87
## pitch_arm           0.00
```


###Applying the Best Model to the ```trainData```


When the best model was applied to ```trainData```, 19 out of 20 cases were correctly predicted, yielding an observed out of sample error rate of:  1-0.95=0.05, or 5%. This agrees well with the estimated out of sample error rate obtained by applying the model to the validation set (95%-97% accuracy; 3%-5% out of sample error rate). (Note that a previous model version built using random forest with leave-one-out cross validation correctly identified the single mis-classified case in ```trainData```, but exhibited overall lower accuracy than the best model ```modFit7``` that was finally selected.)

---


