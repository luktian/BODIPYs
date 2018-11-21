# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:00:57 2018

@author: ly931
"""
### ATTENTION: dataset format: first column should be sample number, the second is target values (PCE) , then the rest are variables

data_set_path = 'dataset.csv'  ###define your data path
data_split_test_percent = 0.2  ###define your testing set percent
feature_number = 5  ###filtered feature number
variables_number = 807  ###variable number in your dataset
ng_threshold = 40  ###GA rounds
population_number = 100 ###Original populations
cross_probability = 1.0 ###probability of crossing
mutate_probability = 0.5 ###probability of mutation

from pandas import read_csv,DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
import random
from deap import base, creator, tools, algorithms
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from functools import partial
from sklearn.metrics import mean_squared_error

######Normalized dataset via MinMax######
data_set_x = read_csv(data_set_path).iloc[:,2:]
descriptor_names = data_set_x.columns.values.tolist()
scaler = MinMaxScaler()
scaler_fit = scaler.fit(data_set_x)
data_set_x = DataFrame(scaler.transform(data_set_x))
data_set_x.columns = descriptor_names
data_set_scaled = concat([read_csv(data_set_path).iloc[:,0:2],data_set_x],axis=1)
del data_set_x, descriptor_names, data_set_path
######Normalized dataset via MinMax######


#######sphere exclusion#######
###calculate the distance
all_descriptor_matrix = data_set_scaled.iloc[:,2:]
all_descriptor_matrix_rownum = all_descriptor_matrix.iloc[:,0].size
distance_matrix = DataFrame(index=range(all_descriptor_matrix_rownum),columns=range(all_descriptor_matrix_rownum))
for i in range(0,all_descriptor_matrix_rownum):
    for j in range(0,all_descriptor_matrix_rownum):
        x = ((((all_descriptor_matrix.iloc[i]-all_descriptor_matrix.iloc[j]))**2).sum(axis=0))**0.5
        distance_matrix.iloc[i,j] = x
distances = DataFrame(index=range(all_descriptor_matrix_rownum),columns=range(2))
for i in range(all_descriptor_matrix_rownum):
    distances.iloc[i,1] = distance_matrix.iloc[:,i].sum(axis=0)/all_descriptor_matrix_rownum
    distances.iloc[i,0] = i + 1
distances = distances.sort_values(by=1).reset_index(drop=True)
###define the interval for the exclusion
exclusion_interval = round(all_descriptor_matrix_rownum/round(all_descriptor_matrix_rownum * data_split_test_percent))
###select the testing samples
test_sample_number = []
for i in range(0, all_descriptor_matrix_rownum-1, exclusion_interval):
    test_sample_number.append(distances.iloc[i,0])
test_set_colunms = []
for i in test_sample_number:
    test_set_colunms.append(data_set_scaled.iloc[i-1,:])
test_set = concat(test_set_colunms,axis=1).T
###select the training samples
for i in range(0, all_descriptor_matrix_rownum-1, exclusion_interval):
     distances.drop([i], inplace=True)
train_sample_number = list(distances.iloc[:,0])
train_set_colunms = []
for i in train_sample_number:
    train_set_colunms.append(data_set_scaled.iloc[i-1,:])
train_set = concat(train_set_colunms,axis=1).T
del all_descriptor_matrix, i, j, x, exclusion_interval, distance_matrix, distances, data_split_test_percent, test_sample_number, train_sample_number, test_set_colunms, train_set_colunms
######sphere exclusion#######


#######GA-MLR########
###define the evaluation mechanism, namely MLR###
def evalOneMax(individual):
    x_dataset_copy = deepcopy(x_dataset)
    dic = {}
    dataset_feature = []
    for i in range(feature_number):
        t = individual[i]
        dic[variables[t]] = t
    for key,value in dic.items():   
        if value in individual:
            dataset_columns = x_dataset_copy.loc[:,key]
            dataset_feature.append(dataset_columns)
    dataset_concat = concat(dataset_feature,axis=1)
    Rmodels = LinearRegression()
    Rmodels.fit(dataset_concat, y_dataset)
    dataset_concat_predictions = Rmodels.predict(dataset_concat)
    mse = mean_squared_error(y_dataset, dataset_concat_predictions)
    return mse ** 0.5,

###define set of train, test, data
train = train_set
x_train = train.iloc[:,2:]
y_train = train.iloc[:,1]
variables = x_train.columns.values.tolist()

test = test_set
x_test = test.iloc[:,2:]
y_test = test.iloc[:,1]

data = data_set_scaled
x_dataset = data.iloc[:,2:]
y_dataset = data.iloc[:,1]



###creating the creators and tools in deap frame
creator.create("FitnessMax", base.Fitness, weights=(-1.0, ))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

#while (trainR2 <= 75 or testR2 <= 75): #since it's not probable to run the GA algorithm only one time, you could define the thresholds of R2. So that the codes would not end until reaching the R2 you want. But attention, the parts involved in the cycle should only contain 'GA parts' and 'check the results of GA'
###the GA parts
gen_idx = partial(random.sample, range(variables_number), feature_number)
toolbox.register("individual", tools.initIterate, creator.Individual, 
        gen_idx)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

pop = toolbox.population(n=population_number)
hof = tools.HallOfFame(1)
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cross_probability, mutpb=mutate_probability, ngen=ng_threshold, 
                                       halloffame=hof, verbose=False) #you could switch whether see the details of GA progress by verbose of 'True' or 'False'

###check the results of GA
dic_hof = {}
x_train_copy_hof = deepcopy(x_train)
train_feature = []
for k in range(feature_number):
    t = hof[0][k]
    dic_hof[variables[t]] = t
for key,value in dic_hof.items():
    if value in hof[0]:
        train_columns = x_train_copy_hof.loc[:,key]
        train_feature.append(train_columns)
train_feature_concat = concat(train_feature,axis=1)
Rmodels = LinearRegression()
Rmodels.fit(train_feature_concat, y_train)
score_hof = Rmodels.score(train_feature_concat,y_train) * 100
test_selected = []
for i in dic_hof.keys():
    n = test.loc[:,i]
    test_selected.append(n)
test_selected = concat(test_selected,axis=1)
score_test = Rmodels.score(test_selected,y_test) * 100
train_predictions = DataFrame(Rmodels.predict(train_feature_concat),columns={'PCE(pred)'})
test_predictions = DataFrame(Rmodels.predict(test_selected),columns={'PCE(pred)'})
train_numandpce = train.iloc[:,0:2].reset_index(drop=True)
test_numandpce = test.iloc[:,0:2].reset_index(drop=True)
train_feature_concat = train_feature_concat.reset_index(drop=True)
test_selected = test_selected.reset_index(drop=True)
filtered_train = concat([train_numandpce, train_predictions, train_feature_concat],axis=1)
filtered_test = concat([test_numandpce, test_predictions, test_selected],axis=1)
trainR2 = score_hof
testR2 = score_test
print("Train R2: {0:.2f}".format(score_hof))
print("Test R2: {0:.2f}".format(score_test))

###For the results compactly, you could del the reduntant code variables below
del all_descriptor_matrix_rownum, cross_probability, data, dic_hof, feature_number, i, k, key, log, mutate_probability, n, ng_threshold, pop, population_number, score_hof, score_test,t, test, test_set, train, train_columns
del train_feature, train_set, value, variables, variables_number,x_dataset, x_test, data_set_scaled, x_train, x_train_copy_hof, y_dataset, y_test, y_train
del test_numandpce, train_numandpce, test_predictions, train_predictions, train_feature_concat, test_selected

###well, you can choose whether output the filtered trainset and testset
filtered_train.to_csv('Filtered_trainset.csv',index=None)
filtered_test.to_csv('Filtered_testset.csv',index=None)