import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor,ExtraTreesRegressor,RandomForestRegressor
import matplotlib.pyplot as plt
from random import randint
from sklearn.model_selection import cross_val_predict

train_data=pd.read_csv('train_data.csv')
test_data=pd.read_csv('test_data.csv')
IDs=test_data['ID']

#get the featuresset created by imputing the missing values
tr1=pd.read_csv('featuresset/tr1.csv')
tr2=pd.read_csv('featuresset/tr2.csv')
tr3=pd.read_csv('featuresset/tr3.csv')
tr4=pd.read_csv('featuresset/tr4.csv')
tr5=pd.read_csv('featuresset/tr5.csv')
tr6=pd.read_csv('featuresset/tr6.csv')

ts1=pd.read_csv('featuresset/ts1.csv')
ts2=pd.read_csv('featuresset/ts2.csv')
ts3=pd.read_csv('featuresset/ts3.csv')
ts4=pd.read_csv('featuresset/ts4.csv')
ts5=pd.read_csv('featuresset/ts5.csv')
ts6=pd.read_csv('featuresset/ts6.csv')


def get_train_test_set(tr,ts):
    ytrain=tr['Footfall']
    xtrain=tr.drop(['Footfall','Unnamed: 0'],1)
    xtest=ts.drop(['Unnamed: 0'],1)
    return(xtrain,ytrain,xtest)

xtrain1,ytrain1,xtest1=get_train_test_set(tr1,ts1)
xtrain2,ytrain2,xtest2=get_train_test_set(tr2,ts2)
xtrain3,ytrain3,xtest3=get_train_test_set(tr3,ts3)
xtrain4,ytrain4,xtest4=get_train_test_set(tr4,ts4)
xtrain5,ytrain5,xtest5=get_train_test_set(tr5,ts5)
xtrain6,ytrain6,xtest6=get_train_test_set(tr6,ts6)


#
params1=params_gbm={'n_estimators':500,'max_depth':4,'learning_rate':0.1,'loss':'ls'}
params2=params_gbm={'n_estimators':1000,'max_depth':5,'learning_rate':0.1,'loss':'ls'}
params3=params_gbm={'n_estimators':1500,'max_depth':4,'learning_rate':0.05,'loss':'ls'}
params4=params_gbm={'n_estimators':1500,'max_depth':5,'learning_rate':0.05,'loss':'ls'}
params5=params_gbm={'n_estimators':1000,'max_depth':4,'learning_rate':0.06,'loss':'ls'}
params6=params_gbm={'n_estimators':1000,'max_depth':5,'learning_rate':0.06,'loss':'ls'}
params7=params_gbm={'n_estimators':800,'max_depth':4,'learning_rate':0.11,'loss':'ls'}
params8=params_gbm={'n_estimators':800,'max_depth':5,'learning_rate':0.11,'loss':'ls'}
params9=params_gbm={'n_estimators':1200,'max_depth':4,'learning_rate':0.07,'loss':'ls'}
params10=params_gbm={'n_estimators':1200,'max_depth':5,'learning_rate':0.07,'loss':'ls'}


#create model objects of gbm
gbm1=GradientBoostingRegressor(**params1,verbose=1,subsample=0.8,random_state=10,max_features='sqrt',min_samples_split=200)
gbm2=GradientBoostingRegressor(**params2,verbose=1,subsample=0.8,random_state=10,max_features='sqrt',min_samples_split=200)
gbm3=GradientBoostingRegressor(**params3,verbose=1,subsample=0.8,random_state=10,max_features='sqrt',min_samples_split=200)
gbm4=GradientBoostingRegressor(**params4,verbose=1,subsample=0.8,random_state=10,max_features='sqrt',min_samples_split=200)
gbm5=GradientBoostingRegressor(**params5,verbose=1,subsample=0.8,random_state=10,max_features='sqrt',min_samples_split=200)
gbm6=GradientBoostingRegressor(**params6,verbose=1,subsample=0.8,random_state=10,max_features='sqrt',min_samples_split=200)
gbm7=GradientBoostingRegressor(**params7,verbose=1,subsample=0.8,random_state=10,max_features='sqrt',min_samples_split=200)
gbm8=GradientBoostingRegressor(**params8,verbose=1,subsample=0.8,random_state=10,max_features='sqrt',min_samples_split=200)
gbm9=GradientBoostingRegressor(**params9,verbose=1,subsample=0.8,random_state=10,max_features='sqrt',min_samples_split=200)
gbm10=GradientBoostingRegressor(**params10,verbose=1,subsample=0.8,random_state=10,max_features='sqrt',min_samples_split=200)


#fit on the approaches
#Not using trainset5 since I imputed missing values of it using gbm
gbm1.fit(xtrain1,ytrain1)
gbm2.fit(xtrain2,ytrain2)
gbm3.fit(xtrain3,ytrain3)
gbm4.fit(xtrain4,ytrain4)
gbm5.fit(xtrain6,ytrain6)
gbm6.fit(xtrain1,ytrain1)
gbm7.fit(xtrain2,ytrain2)
gbm8.fit(xtrain3,ytrain3)
gbm9.fit(xtrain4,ytrain4)
gbm10.fit(xtrain6,ytrain6)


def predict_test_set(model,test_set,IDs):
    output=pd.DataFrame()
    output['Footfall']=model.predict(test_set)
    output['ID']=IDs
    return(output['Footfall'])


gbm1_preds=predict_test_set(gbm1,xtest1,IDs)
gbm2_preds=predict_test_set(gbm2,xtest2,IDs)
gbm3_preds=predict_test_set(gbm3,xtest3,IDs)
gbm4_preds=predict_test_set(gbm4,xtest4,IDs)
gbm5_preds=predict_test_set(gbm5,xtest6,IDs)
gbm6_preds=predict_test_set(gbm6,xtest1,IDs)
gbm7_preds=predict_test_set(gbm7,xtest2,IDs)
gbm8_preds=predict_test_set(gbm8,xtest3,IDs)
gbm9_preds=predict_test_set(gbm9,xtest4,IDs)
gbm10_preds=predict_test_set(gbm10,xtest6,IDs)


#After submitting we got to know that the model 9 is the best model so fitting it on all testset except 5

gbm19=gbm9.fit(xtrain1,ytrain1)
gbm29=gbm9.fit(xtrain2,ytrain2)
gbm39=gbm9.fit(xtrain3,ytrain3)
gbm49=gbm9.fit(xtrain4,ytrain4)
gbm59=gbm9.fit(xtrain6,ytrain6)


gbm19_preds=predict_test_set(gbm19,xtest1,IDs)
gbm29_preds=predict_test_set(gbm29,xtest2,IDs)
gbm39_preds=predict_test_set(gbm39,xtest3,IDs)
gbm49_preds=predict_test_set(gbm49,xtest4,IDs)
gbm59_preds=predict_test_set(gbm59,xtest6,IDs)


#averaging the results from best model
gbm9_avg_preds=(gbm19_preds + gbm29_preds + gbm39_preds + gbm49_preds + gbm59_preds)/5
output=pd.DataFrame()
output['Footfall']=gbm9_avg_preds
output['ID']=IDs

#plot the important features considered by GBM
imp_feats=pd.Series(gbm9.feature_importances_,xtrain1.columns).sort_values(ascending=False)
imp_feats.plot(kind='bar',title='Feature Importances')
plt.ylabel('Feature Importance score')


# ### Using Random Forest Regressor

#initialise RF regressor
rf1=RandomForestRegressor(max_depth=4,max_features='sqrt',random_state=10,n_estimators=200)
rf2=RandomForestRegressor(max_depth=5,max_features='sqrt',random_state=10,n_estimators=200)
rf3=RandomForestRegressor(max_depth=4,max_features='sqrt',random_state=10,n_estimators=200)
rf4=RandomForestRegressor(max_depth=5,max_features='sqrt',random_state=10,n_estimators=200)
rf5=RandomForestRegressor(max_depth=4,max_features='sqrt',random_state=10,n_estimators=200)


#Not including dataset in which I have imputed missing values using Random Forest
rf1_out=rf1.fit(xtrain1,ytrain1)
rf2_out=rf2.fit(xtrain2,ytrain2)
rf3_out=rf3.fit(xtrain3,ytrain3)
rf4_out=rf5.fit(xtrain5,ytrain5)
rf5_out=rf6.fit(xtrain6,ytrain6)


#predict
rf1_preds=predict_test_set(rf1_out,xtest1,IDs)
rf2_preds=predict_test_set(rf2_out,xtest2,IDs)
rf3_preds=predict_test_set(rf3_out,xtest3,IDs)
rf4_preds=predict_test_set(rf4_out,xtest5,IDs)
rf5_preds=predict_test_set(rf5_out,xtest6,IDs)


# averaging the results and checking the scores
rf_preds_avg=(rf1_preds+rf2_preds+rf3_preds+rf4_preds+rf5_preds)/5
rf_output=pd.DataFrame()
rf_output['Footfall']=rf_preds_avg
rf_output['ID']=IDs


#avg results from the random forest are giving worst performance lets avg out gbm and rf output to check results
#first I will choose all the gbm models predictions and check , then 
rf_gbm_preds_avg=(gbm1_preds+gbm2_preds+gbm3_preds+gbm4_preds+gbm5_preds+gbm6_preds+gbm7_preds+gbm8_preds+gbm9_preds+
                 rf1_preds+rf2_preds+rf3_preds+rf4_preds+rf5_preds)/14
rf_gbm=pd.DataFrame()
rf_gbm['Footfall']=rf_gbm_preds_avg
rf_gbm['ID']=IDs


#choose best performing models of gbm and rf and then add them and check score
rf_gbm_bm_preds_avg=(gbm4_preds+gbm5_preds+gbm9_preds+rf3_preds+rf5_preds)/5
rf_gbm_bm=pd.DataFrame()
rf_gbm_bm['Footfall']=rf_gbm_bm_preds_avg
rf_gbm_bm['ID']=IDs


# #### Extra Tree Regressor


#trying extra tree regressor to check if it gives better performance than gbm
#Code reference: Machine Learning mastery ExtraTreesClassifier

ext1=ExtraTreesRegressor(max_depth=4,max_features='sqrt',random_state=10,n_estimators=200)
ext2=ExtraTreesRegressor(max_depth=5,max_features='sqrt',random_state=10,n_estimators=200)
ext3=ExtraTreesRegressor(max_depth=4,max_features='sqrt',random_state=10,n_estimators=200)
ext4=ExtraTreesRegressor(max_depth=5,max_features='sqrt',random_state=10,n_estimators=200)
ext5=ExtraTreesRegressor(max_depth=4,max_features='sqrt',random_state=10,n_estimators=200)
ext6=ExtraTreesRegressor(max_depth=4,max_features='sqrt',random_state=10,n_estimators=200)


#fit on the data
ext1_model=ext1.fit(xtrain1,ytrain1)
ext2_model=ext2.fit(xtrain2,ytrain2)
ext3_model=ext3.fit(xtrain3,ytrain3)
ext4_model=ext4.fit(xtrain4,ytrain4)
ext5_model=ext5.fit(xtrain5,ytrain5)
ext6_model=ext6.fit(xtrain6,ytrain6)


#predict 
ext1_preds=predict_test_set(ext1_model,xtest1,IDs)
ext2_preds=predict_test_set(ext2_model,xtest2,IDs)
ext3_preds=predict_test_set(ext3_model,xtest3,IDs)
ext4_preds=predict_test_set(ext4_model,xtest4,IDs)
ext5_preds=predict_test_set(ext5_model,xtest5,IDs)
ext6_preds=predict_test_set(ext6_model,xtest6,IDs)


#checking the score by combining output of all models (still low score 139)
ext_rf_gbm_avg=(gbm1_preds+gbm2_preds+gbm3_preds+gbm4_preds+gbm5_preds+gbm6_preds+gbm7_preds+gbm8_preds+gbm9_preds+
               rf1_preds+rf2_preds+rf3_preds+rf4_preds+rf5_preds+ext1_preds+ext2_preds+ext3_preds+ext4_preds+ext5_preds
               +ext6_preds)/20

ext_rf_gbm=pd.DataFrame()
ext_rf_gbm['Footfall']=ext_rf_gbm_avg
ext_rf_gbm['ID']=IDs


# In[56]:


#combining preds of best models 
ext_rf_gbm_bm_avg=(gbm4_preds+gbm5_preds+gbm9_preds+rf3_preds+rf5_preds+ext3_preds+ext5_preds)/7
ext_rf_gbm_bm=pd.DataFrame()
ext_rf_gbm_bm['Footfall']=ext_rf_gbm_bm_avg
ext_rf_gbm_bm['ID']=IDs


# ### Stacking the ensemble
#to check if we get better LB score

#refering the sample code from analytics vidhya
models=[gbm1,gbm2,gbm3,gbm4,gbm5,gbm6,gbm7,gbm8,gbm9,gbm10,rf3,rf5,ext2]
model_names=['gbm1','gbm2','gbm3','gbm4','gbm5','gbm6','gbm7','gbm8','gbm9','gbm10','rf3','rf5','ext2']


train_stack=pd.DataFrame()
test_stack=pd.DataFrame()

train_stack=[xtrain1,xtrain2,xtrain3,xtrain4,xtrain5,xtrain6]
train_labels_stack=[ytrain1,ytrain2,ytrain3,ytrain4,ytrain5,ytrain6]
test_stack=[xtest1,xtest2,xtest3,xtest4,xtest5,xtest6]


#choose a model at time and randomly choose the test and train set
train_set=pd.DataFrame()
test_set=pd.DataFrame()
for (model,model_name) in zip(models,model_names):
    r=randint(0,5)
    print(r)
    train_set[model_name]=cross_val_predict(model,train_stack[r],train_labels_stack[r],cv=5)
    model.fit(train_stack[r],train_labels_stack[r])
    test_set[model_name]=model.predict(test_stack[r])
    

train_set.to_csv('stacking_train.csv')
test_set.to_csv('stacking_test.csv')


print(train_set.head(5))


print(test_set.head(5))


#the predictions of the models stacked together and formed as dataset. We will fit gbm on this dataset 
#check the final score LB 129
params_stack={'n_estimators':200,'max_depth':5,'learning_rate':0.02,'loss':'ls'}
gbm_stack=GradientBoostingRegressor(**params_stack,verbose=1,subsample=0.8,random_state=10,max_features='sqrt',min_samples_split=200)
gbm_stack.fit(train_set,ytrain1)
ensemble_out1=predict_test_set(gbm_stack,test_set,IDs)

