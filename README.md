# BODIPYs
The GA-MLR method for the BODIPY dyes

I'm supposing that you have read my paper about BODIPY dyes for DSSCs (Not published untill 2018.11.21.). So, the details about how to use it is listed as below and you could also be familiar with it in the supporting information of my paper.


Illustration: In this code, user should input the path of dataset and define the parameters listed at the beginning. The dataset would be normalized via the method of minmax scaler in sklearn library1. Then the data would be divided into training set and testing set with ratio that user defined above based on the technology of sphere-exclusion2 which wrote by ourselves. For the dataset, the genetic algorithm would be used based on the DEAP framework4 while the evaluation mechanism is the multiple linear regression (MLR) and mean squared error (MSE) which are quoted from sklearn library1. As for the features selected through GA, the coefficient of determinations for training set and testing are calculated to ensure the reliability of models. Other parameters are not supplied in this code but easy to be reproduced since the formulas are presented clearly in Table S5.
