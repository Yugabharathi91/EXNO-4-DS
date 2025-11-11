# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file...

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
 import pandas as pd
 import numpy as np
 import seaborn as sns
 from sklearn.model_selection import train_test_split
 from sklearn.neighbors import KNeighborsClassifier
 from sklearn.metrics import accuracy_score, confusion_matrix
 data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
 data
```

<img width="1620" height="654" alt="image" src="https://github.com/user-attachments/assets/106becc5-5fcd-429c-82e1-00695861e8f5" />

```
 data.isnull().sum()
```

<img width="210" height="583" alt="image" src="https://github.com/user-attachments/assets/222d03f9-e799-4126-953a-9d2eeb4db585" />

```
 missing=data[data.isnull().any(axis=1)]
 missing
```

<img width="1476" height="517" alt="image" src="https://github.com/user-attachments/assets/44d2cb89-a895-4c6a-b6fd-039d78bd7f53" />

```
data2=data.dropna(axis=0)
data2
```

<img width="1612" height="515" alt="image" src="https://github.com/user-attachments/assets/ebd3bb1b-4e6c-41f0-a8f7-de6dd9a6f977" />

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

<img width="1223" height="450" alt="image" src="https://github.com/user-attachments/assets/12b258ca-7486-4b13-934d-6a36f52cb0c8" />

```
 sal2=data2['SalStat']
 dfs=pd.concat([sal,sal2],axis=1)
 dfs
```

<img width="451" height="551" alt="image" src="https://github.com/user-attachments/assets/9687c83f-d36a-40f3-af1c-91c7a919f395" />

```
data2
```

<img width="1468" height="503" alt="image" src="https://github.com/user-attachments/assets/1ba77e7c-8dbc-461f-a993-e2222f5117d4" />

```
 new_data=pd.get_dummies(data2, drop_first=True)
 new_data
```

<img width="1755" height="608" alt="image" src="https://github.com/user-attachments/assets/da030337-02de-4b38-8579-50fdabf81e84" />

```
columns_list=list(new_data.columns)
print(columns_list)
```
```
['age', 'capitalgain', 'capitalloss', 'hoursperweek', 'SalStat', 'JobType_ Local-gov', 'JobType_ Private', 'JobType_ Self-emp-inc', 'JobType_ Self-emp-not-inc', 'JobType_ State-gov', 'JobType_ Without-pay', 'EdType_ 11th', 'EdType_ 12th', 'EdType_ 1st-4th', 'EdType_ 5th-6th', 'EdType_ 7th-8th', 'EdType_ 9th', 'EdType_ Assoc-acdm', 'EdType_ Assoc-voc', 'EdType_ Bachelors', 'EdType_ Doctorate', 'EdType_ HS-grad', 'EdType_ Masters', 'EdType_ Preschool', 'EdType_ Prof-school', 'EdType_ Some-college', 'maritalstatus_ Married-AF-spouse', 'maritalstatus_ Married-civ-spouse', 'maritalstatus_ Married-spouse-absent', 'maritalstatus_ Never-married', 'maritalstatus_ Separated', 'maritalstatus_ Widowed', 'occupation_ Armed-Forces', 'occupation_ Craft-repair', 'occupation_ Exec-managerial', 'occupation_ Farming-fishing', 'occupation_ Handlers-cleaners', 'occupation_ Machine-op-inspct', 'occupation_ Other-service', 'occupation_ Priv-house-serv', 'occupation_ Prof-specialty', 'occupation_ Protective-serv', 'occupation_ Sales', 'occupation_ Tech-support', 'occupation_ Transport-moving', 'relationship_ Not-in-family', 'relationship_ Other-relative', 'relationship_ Own-child', 'relationship_ Unmarried', 'relationship_ Wife', 'race_ Asian-Pac-Islander', 'race_ Black', 'race_ Other', 'race_ White', 'gender_ Male', 'nativecountry_ Canada', 'nativecountry_ China', 'nativecountry_ Columbia', 'nativecountry_ Cuba', 'nativecountry_ Dominican-Republic', 'nativecountry_ Ecuador', 'nativecountry_ El-Salvador', 'nativecountry_ England', 'nativecountry_ France', 'nativecountry_ Germany', 'nativecountry_ Greece', 'nativecountry_ Guatemala', 'nativecountry_ Haiti', 'nativecountry_ Holand-Netherlands', 'nativecountry_ Honduras', 'nativecountry_ Hong', 'nativecountry_ Hungary', 'nativecountry_ India', 'nativecountry_ Iran', 'nativecountry_ Ireland', 'nativecountry_ Italy', 'nativecountry_ Jamaica', 'nativecountry_ Japan', 'nativecountry_ Laos', 'nativecountry_ Mexico', 'nativecountry_ Nicaragua', 'nativecountry_ Outlying-US(Guam-USVI-etc)', 'nativecountry_ Peru', 'nativecountry_ Philippines', 'nativecountry_ Poland', 'nativecountry_ Portugal', 'nativecountry_ Puerto-Rico', 'nativecountry_ Scotland', 'nativecountry_ South', 'nativecountry_ Taiwan', 'nativecountry_ Thailand', 'nativecountry_ Trinadad&Tobago', 'nativecountry_ United-States', 'nativecountry_ Vietnam', 'nativecountry_ Yugoslavia']
```

```
 features=list(set(columns_list)-set(['SalStat']))
 print(features)
```

```
['maritalstatus_ Married-spouse-absent', 'EdType_ 7th-8th', 'JobType_ Self-emp-not-inc', 'occupation_ Priv-house-serv', 'maritalstatus_ Never-married', 'capitalgain', 'nativecountry_ Mexico', 'nativecountry_ Nicaragua', 'hoursperweek', 'maritalstatus_ Married-AF-spouse', 'occupation_ Transport-moving', 'race_ Asian-Pac-Islander', 'relationship_ Other-relative', 'nativecountry_ Germany', 'nativecountry_ Peru', 'nativecountry_ Canada', 'occupation_ Craft-repair', 'nativecountry_ Trinadad&Tobago', 'nativecountry_ United-States', 'nativecountry_ China', 'JobType_ Without-pay', 'nativecountry_ Yugoslavia', 'race_ Other', 'race_ Black', 'nativecountry_ Honduras', 'nativecountry_ Guatemala', 'EdType_ Assoc-acdm', 'age', 'EdType_ Preschool', 'race_ White', 'EdType_ Some-college', 'nativecountry_ Hungary', 'nativecountry_ Holand-Netherlands', 'nativecountry_ Italy', 'nativecountry_ Ireland', 'nativecountry_ Japan', 'nativecountry_ Taiwan', 'occupation_ Handlers-cleaners', 'JobType_ Private', 'relationship_ Not-in-family', 'nativecountry_ El-Salvador', 'occupation_ Machine-op-inspct', 'nativecountry_ Iran', 'nativecountry_ Cuba', 'occupation_ Tech-support', 'nativecountry_ Haiti', 'nativecountry_ France', 'nativecountry_ Philippines', 'nativecountry_ Outlying-US(Guam-USVI-etc)', 'occupation_ Other-service', 'occupation_ Armed-Forces', 'nativecountry_ Jamaica', 'nativecountry_ Hong', 'relationship_ Unmarried', 'nativecountry_ Poland', 'nativecountry_ Greece', 'EdType_ Prof-school', 'occupation_ Sales', 'EdType_ 9th', 'JobType_ State-gov', 'EdType_ Assoc-voc', 'occupation_ Exec-managerial', 'relationship_ Wife', 'occupation_ Farming-fishing', 'maritalstatus_ Separated', 'JobType_ Self-emp-inc', 'nativecountry_ South', 'relationship_ Own-child', 'nativecountry_ Portugal', 'EdType_ 11th', 'EdType_ Bachelors', 'gender_ Male', 'maritalstatus_ Widowed', 'nativecountry_ Columbia', 'capitalloss', 'nativecountry_ India', 'nativecountry_ Dominican-Republic', 'occupation_ Protective-serv', 'EdType_ Doctorate', 'nativecountry_ Puerto-Rico', 'EdType_ HS-grad', 'EdType_ 1st-4th', 'JobType_ Local-gov', 'occupation_ Prof-specialty', 'EdType_ 5th-6th', 'EdType_ Masters', 'maritalstatus_ Married-civ-spouse', 'nativecountry_ Vietnam', 'nativecountry_ Laos', 'nativecountry_ Scotland', 'nativecountry_ England', 'EdType_ 12th', 'nativecountry_ Ecuador', 'nativecountry_ Thailand'
```
```
 y=new_data['SalStat'].values
 print(y)
```
<img width="288" height="93" alt="image" src="https://github.com/user-attachments/assets/f0f2d539-1d61-4218-9475-89e7cedf548d" />

```
 x=new_data[features].values
 print(x)
```

<img width="480" height="214" alt="image" src="https://github.com/user-attachments/assets/b3ab5ec9-0431-4bda-9558-bd7d21ae118d" />

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
 KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
 KNN_classifier.fit(train_x,train_y)
```

<img width="787" height="170" alt="image" src="https://github.com/user-attachments/assets/6b9e7062-b221-43bd-a5bb-09ab351d452b" />

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

<img width="509" height="139" alt="image" src="https://github.com/user-attachments/assets/100e28ad-28b9-469a-9503-88237a1829a3" />

```
accuracy_score=accuracy_score(test_y,prediction)
 print(accuracy_score)
```

<img width="436" height="88" alt="image" src="https://github.com/user-attachments/assets/13c81945-cccd-4b5f-8bb4-683082dd8ec3" />


```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```

<img width="607" height="80" alt="image" src="https://github.com/user-attachments/assets/13e3c3ba-515a-4541-8901-bea06d322b26" />

```
data.shape
```

<img width="155" height="71" alt="image" src="https://github.com/user-attachments/assets/06865aef-0062-41c3-9500-1af6aa4a6c92" />

```
import pandas as pd
 from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
 data={
 'Feature1': [1,2,3,4,5],
 'Feature2': ['A','B','C','A','B'],
 'Feature3': [0,1,1,0,1],
 'Target'  : [0,1,1,0,1]
 }
 df=pd.DataFrame(data)
 x=df[['Feature1','Feature3']]
 y=df[['Target']]
 selector=SelectKBest(score_func=mutual_info_classif,k=1)
 x_new=selector.fit_transform(x,y)
 selected_feature_indices=selector.get_support(indices=True)
 selected_features=x.columns[selected_feature_indices]
 print("Selected Features:")
 print(selected_features)
```

<img width="904" height="474" alt="image" src="https://github.com/user-attachments/assets/e10878fc-c098-4e54-b3e1-f65bfbe8f134" />

```
 import pandas as pd
 import numpy as np
 from scipy.stats import chi2_contingency
 import seaborn as sns
 tips=sns.load_dataset('tips')
 tips.head()
```

<img width="561" height="373" alt="image" src="https://github.com/user-attachments/assets/8dad4d47-0247-4f59-adc9-a7afabbb8931" />

```
tips.time.unique()
```

<img width="446" height="97" alt="image" src="https://github.com/user-attachments/assets/2a9a4f79-ddde-42bb-bfe4-731ab96bfdff" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
 print(contingency_table)
```

<img width="529" height="154" alt="image" src="https://github.com/user-attachments/assets/f789a4bb-5009-4bb9-bf3c-890ec0b5cb06" />

```
 chi2,p,_,_=chi2_contingency(contingency_table)
 print(f"Chi-Square Statistics: {chi2}")
 print(f"P-Value: {p}")
```


<img width="444" height="135" alt="image" src="https://github.com/user-attachments/assets/860bd967-fd2c-4008-b0c9-d3276d78362a" />


# RESULT:
       # Thus, To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file is successfully completed.

