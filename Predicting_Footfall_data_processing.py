
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

#Read train and test data
train_data=pd.read_csv('train_data.csv')
test_data=pd.read_csv('test_data.csv')
print(train_data.columns)
print(train_data.head(5))


#Split date into day,month and year in train and test
train_date_obj=pd.DatetimeIndex(train_data['Date'])
train_data['year']=train_date_obj.year
train_data['month']=train_date_obj.month
train_data['day']=train_date_obj.day


#Similarly for the test data
test_date_obj=pd.DatetimeIndex(test_data['Date'])
test_data['year']=test_date_obj.year
test_data['month']=test_date_obj.month
test_data['day']=test_date_obj.day


#Remove the id column from the test and train since its unecessary
#Also drop the date column
IDs=test_data.ID
train_data=train_data.drop('ID',1)
train_data=train_data.drop('Date',1)
test_data=test_data.drop('ID',1)
test_data=test_data.drop('Date',1)
print(train_data.head(5))




#check for the null values present in every column in train
print(train_data.isnull().sum())




#similary for the null values present in every column in test
print(test_data.isnull().sum())


# #### Since large number of missing values are there we cannot afford deleting them. Hence we can have approaches to predict the missing values


def fill_all_missing_using_categorical_features(data,model):
    main_data=data.copy()
    NA_cols=main_data.loc[:,pd.isnull(main_data).sum()>0].columns
    for missing_features in NA_cols:
        Non_NA_features=["year","month","Location_Type","Park_ID","day"]
        
        if main_data[missing_features].isnull().sum()==0:
            break
        print("Finding missing values for",missing_features)
        tr_data=main_data[~main_data[missing_features].isnull()]
        ts_data=main_data[main_data[missing_features].isnull()]
        xtrain=tr_data[Non_NA_features]
        ytrain=tr_data[missing_features]
        xtest=ts_data[Non_NA_features]
        model.fit(xtrain,ytrain)
        
        main_data.loc[main_data[missing_features].isnull(),missing_features]=model.predict(xtest)
    return(main_data)
    
    

def fill_missing_values_avg(data):
    main_data=data.copy()
    NA_cols=main_data.loc[:,pd.isnull(main_data).sum()>0].columns
    for col_name in NA_cols:
        if(main_data[col_name].isnull().sum()==0):
            break
        main_data[col_name]=main_data.groupby(["year","month","Location_Type"])[col_name].transform(lambda x:x.fillna(x.mean()))
    return main_data
   
        
        

#Simple groupby using year,month and Location_Type and then averaging

tr1=fill_missing_values_avg(train_data)
ts1=fill_missing_values_avg(test_data)

print("Missing values now in train",tr1.isnull().sum().sum(),"and in test",ts1.isnull().sum().sum())

#Save features1
tr1.to_csv('featuresset/tr1.csv')
ts1.to_csv('featuresset/ts1.csv')


#groupby month since footfalls depend on months
tr2=train_data.groupby(['month']).transform(lambda x:x.fillna(x.mean()))
ts2=test_data.groupby(['month']).transform(lambda x:x.fillna(x.mean()))

tr2['month']=train_data['month']
ts2['month']=test_data['month']

print(tr2.head(10))
print(ts2.head(10))

print('Missing values in train',tr2,'in test is',ts2)

#save this feature set
tr2.to_csv('featuresset/tr2.csv')
ts2.to_csv('featuresset/ts2.csv')



#Imputing the missing values by predicting them using KNN
knn_reg=KNeighborsRegressor(n_neighbors=5)

#Impute
tr3=fill_all_missing_using_categorical_features(train_data,knn_reg)
ts3=fill_all_missing_using_categorical_features(test_data,knn_reg)


print('Missing values in train',tr3.isnull().sum().sum(),'in test is',ts3.isnull().sum().sum())

tr3.to_csv('featuresset/tr3.csv')
ts3.to_csv('featuresset/ts3.csv')



#Imputing the values using random forest regressor
rfr=RandomForestRegressor(n_estimators=70)

tr4=fill_all_missing_using_categorical_features(train_data,rfr)
ts4=fill_all_missing_using_categorical_features(test_data,rfr)

print('Missing values in train',tr4.isnull().sum().sum(),'in test is',ts4.isnull().sum().sum())

tr4.to_csv('featuresset/tr4.csv')
ts4.to_csv('featuresset/ts4.csv')


#Using gradient boosting to impute missing values 
gbmr=GradientBoostingRegressor(n_estimators=200,learning_rate=0.2,max_depth=4,min_samples_split=1.0)

tr5=fill_all_missing_using_categorical_features(train_data,gbmr)
ts5=fill_all_missing_using_categorical_features(test_data,gbmr)

print('Missing values in train',tr5.isnull().sum().sum(),'in test is',ts5.isnull().sum().sum())

tr5.to_csv('featuresset/tr5.csv')
ts5.to_csv('featuresset/ts5.csv')



#Average of all the ways which we imputed the missing values to create a final set of 
tr6=(tr1+tr2+tr3+tr4+tr5)/5
ts6=(ts1+ts2+ts3+ts4+ts5)/5

print('Missing values in train',tr6.isnull().sum().sum(),'in test is',ts6.isnull().sum().sum())

tr6.to_csv('featuresset/tr6.csv')
ts6.to_csv('featuresset/ts6.csv')

