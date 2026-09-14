#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


# In[2]:


data = "C:/Users/Asus/OneDrive/Desktop/Projects/Customer_Churn/churn_db.csv"


# In[3]:


df= pd.read_csv(data)


# In[4]:


df.columns = df.columns.str.replace(' ', '_').str.lower()

categorical_columns = list(df.select_dtypes('object').columns)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')


# In[5]:


df


# In[6]:


# tc = pd.to_numeric(df.totalcharges, errors='coerce')


# In[7]:


#converted the totalcharges into numeric

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')


# In[8]:


#filled null values within totalcharges
df.totalcharges = df.totalcharges.fillna(0)


# In[9]:


#converted target variable (churn) into numeric 1 and 0

df.churn = (df.churn == 'yes').astype('int')


# # Setting up Validation framework

# In[10]:


#importing the train test library
from sklearn.model_selection import train_test_split


# In[11]:


#Firstly we split the daya into a full_train and test splits. With 80% of the data being for full train and 20% of the data for testing
# We created full test dataset to train the model after validation

df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state=1)


# In[12]:


#verifying the 80:20 ratio

len(df_full_train), len(df_test)


# In[13]:


#Now we further split the full_train data into train and validation datasets with training data being 75% andvalidation data being 25% of the full train data
# We use validation dataset to validate different models and their accuracies

df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state=1)


# In[14]:


#verifying the dataset lengths

len(df_train), len(df_val), len(df_test)


# In[15]:


#reseting the indecies on these datasets

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[16]:


#isolating target variable vectors or actual values that needs tobe predicted seperately to further verify the predictions

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values 


# In[17]:


#removing the target variable from the training and testing datasets

del df_train['churn']
del df_val['churn']
del df_test['churn']


# # EDA 

# In[18]:


#doing some exploratory analysis on the full train data
#NOTE: we have not removed the target variable from this dataset to learn about the correlation and mutual informations that would be further important to know which independent variables are more important
#      to predict the target variable

df_full_train = df_full_train.reset_index(drop=True)


# In[19]:


#checking how many churners are there in the whole datasest (train + val)

df_full_train.churn.value_counts(normalize=True) #Normalize just converts the numbers into percentages


# In[20]:


#global churn rate is the rate at which people are churning.
#By this data we can conclude that 27% of all curtomers are churning 
#NOTE: golbal churn rate and total churn count will be eventaully the same as we are dealing with 0 and 1 so all the 0 values are already neglicted in both the calculations

global_churn_rate = df_full_train.churn.mean()
round(global_churn_rate, 2)


# In[21]:


#creating a seperate list of numerical coulmns

df_full_train.select_dtypes(['int', 'float'])
numerical = ['tenure', 'monthlycharges', 'totalcharges']


# In[22]:


#creating a seperate list of categorical coulmns

df_full_train.select_dtypes('object').columns
categorical = ['gender', 'partner', 'dependents', 'phoneservice',
       'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup',
       'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies',
       'contract', 'paperlessbilling', 'paymentmethod']


# In[23]:


#checking how many unique values are there within each categorical column
#eg gender = 2 (male and female)

df_full_train[categorical].nunique()


# # Feature Importance

# ## Churn Rate and Risk Ratio

# In[24]:


#here we are analyzing the imporatance of gender in terms of churn prediction

# Here we are checking how many females churned out of the entire population
df_full_train[df_full_train.gender == 'female'].churn.mean()


# In[25]:


# We do the same for males
df_full_train[df_full_train.gender == 'male'].churn.mean()


# In[26]:


global_churn_rate


# ### Difference
# 
# Difference = Global Rate - Group Rate
# 
# - If the difference is positive --> Group is more likely to churn
# - If the difference is negative --> Group is less likely to churn

# In[27]:


# Difference is basically what will happend if we remove the group rate with the global rate
# group rate is what we calculated above with males and females
# Here we are checking if having a partner affects your churning rate or not

#Witha partner your churn rate is 20.5%
churn_partner = df_full_train[df_full_train.partner == 'yes'].churn.mean()

global_churn_rate - churn_partner


# In[28]:


#But without a partner that rate increases to 32.9%

#With this info we can alreadyconclude that people without a partner are more likely to churn than people with a partner
churn_no_partner = df_full_train[df_full_train.partner == 'no'].churn.mean()
global_churn_rate - churn_no_partner


# ### Risk Ratio
# 
# Risk = Group / Global
# 
# - If the risk > 1 --> Group is more likely to churn
# - If the risk < 1 --> Group is less likely to churn

# In[29]:


#we can calculate the risk of that groupto check how likely a person is to churn if they fall under that group
# Above we calculated churn_partner_rate and churn_no_partner_rate we divide them individually with the global_churn_rate

# 0.329 / 0.269
churn_no_partner / global_churn_rate

# here the risk is > 1 which means the numerator is greater than the denominator
# Which means churn_no_partner group is morelikely to churn


# In[30]:


# 0.205 / 0.269 
churn_partner / global_churn_rate

# We can observe the contrast when we check the risk of churn_partner group
# the risk here is < 1 which means the churn_partner rate is smaller than the global_churn_rate
# making this group less likely to churn


# #### SQL QUERY TO FIND THE CHURN RATE AND RISK RATIO FOR EVERY CATEGORICAL DATA
# 
# ```
# SELECT
#     gender,
#     AVG(churn),
#     AVG(churn) - global_churn AS diff
#     AVG(churn) / global_churn AS risk
# 
# FROM
#     data
# 
# GROUP BY
#     gender
# ```

# In[ ]:





# In[31]:


#Pandas translation of the query

#Now we use this pandas query to check the difference and risk for each column and their unique values
for c in categorical:
    df_group = df_full_train.groupby(c).churn.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn_rate
    df_group['risk'] = df_group['mean'] / global_churn_rate
    display(df_group)
    print()
    print()


# ## Mutual Information
# 
# Concept from information theory, it tells us how muc we can learn about one variable if we know value of another

# In[ ]:





# In[32]:


#very similar to correlation for numeric variables 
mutual_info_score(df_full_train.churn, df_full_train.streamingmovies)


# In[33]:


mutual_info_score(df_full_train.churn, df_full_train.contract)


# In[34]:


mutual_info_score(df_full_train.churn, df_full_train.gender)


# In[35]:


# for c in categorical:
#     score = mutual_info_score(df_full_train.churn, df_full_train[c])
#     print(score, c)
#     print()


# In[36]:


def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)


# In[37]:


# This tells us which categories are the most mutually related to the growth of churn rate
mi = df_full_train[categorical].apply(mutual_info_churn_score)
mi.sort_values(ascending = False)


# ## Correlation
# Mutual Information but for numerical data: Correlation Coefficient

# In[38]:


df_full_train.tenure.max()


# In[39]:


df_full_train[numerical].corrwith(df_full_train.churn)
# here a positive coefficient means the growth of that numerical variable is directly proportional to the growth of churn rate
# If that variable increases the churn rate increases
# If its negative then it is inversely proportional to the target variable


# # One-Hot Encoding

# In[ ]:





# In[40]:


#in one-hot encoding we are just converting the categorical data into numerical data
#for eg if gender is male then the gender column will have values 1 and if not then gender column will have 0.
# if there are more than 3 unique variables then for each new variable a new column is created with values 1 wherever that variable is present or true

train_dicts = df_train[categorical + numerical].to_dict(orient = 'records')


# In[41]:


train_dicts


# In[42]:


dv = DictVectorizer(sparse=False) #without sparse it forms a sparse matrix

#DictVectorizer by default stores the matrix as a sparse matrix
# A sparse matrix only stores the inofmration about where the 1 value is stored withing the dense matrix

#Eg [row, column, value]
#   [ 1 ,   2   ,  1   ]

#It tells us that there is a value 1 which is at row 1 col 2 of the dense matrix
# This saves a lot of memory as we are not storing the unncessary zeroes


# In[43]:


#Creating X_train array we'll use to train the model
X_train = dv.fit_transform(train_dicts)
X_train


# In[44]:


# dv.get_feature_names_out()


# In[45]:


val_dicts = df_val[categorical + numerical].to_dict(orient = 'records')


# In[46]:


#Same for validation
X_val = dv.transform(val_dicts)


# # Logistic Regression

# In[47]:


#Logistic function is nothing but squishification of linear regression values but its not that simple
#Instead of just converting the linear regression values between 0 and 1. We find out the probability of the prediction being 0 and 1
# If the probability of it being 1 is 0.7 and prob of it bein 0 is 0.3. Then based on our threshold we can identify in which category we will classify this prediction


#We squishify the values using sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[48]:


z = np.linspace(-7,5,51)


# In[49]:


z


# In[50]:


sigmoid(z)


# In[51]:


plt.plot(z, sigmoid(z))


# In[52]:


def linear_regression(xi):
    res = w0

    for j in range(len(xi)):
        res = res + (xi[i] * w[i])

    return res


# In[53]:


def logistic_regression(xi):
    res = w0

    for j in range(len(xi)):
        res = res + (xi[i] * w[i])

    result = sigmoid(res)
    return result



# # Training Logistic Regression

# In[ ]:





# In[54]:


#Here instead of manual logistic regression model we will be using ths sklearn logistic regression which is more accurate
model = LogisticRegression()
model.fit(X_train, y_train)


# In[55]:


model.intercept_[0]


# In[56]:


model.coef_[0].round(3)


# In[57]:


#These are the hard predictions which means direct converted values
model.predict(X_train) #Hard Predictions

# These are prediction probabilities which tells us both the probabilities
#Prob of being 0
#Prob of being 1
y_pred = model.predict_proba(X_val)[:, 1] #Soft Predictions


# In[58]:


#This is important as this is the threshold we'll be using to classify a particular prediction into churn or no_churn (1 or 0)
#Eg. If predcition prob of a customer being a churner (1) is 0.64 then we'll classify that customer as 1 as there is a 64%chance that this customer will churn

churn_decision = (y_pred >= 0.5)


# In[59]:


#All these customers are more likely to churn

df_val[churn_decision].customerid


# ### Finding Accuracy

# In[60]:


#Finding how many target variables we correctly predicted

(y_val == churn_decision).sum()


# ## Finding the best threshold

# In[61]:


# This is important as most of our predictions are dependent on our decision boundary or the threshold
# We check the accuracy score with each and every possible threshold to see which one is giving us the highest accuracy 

thresholds = np.linspace(0,1,21)

scores = []

for t in thresholds:
    churn_decision = (y_pred>= t)
    score = (y_val == churn_decision).mean()
    scores.append(score)


# In[62]:


#here we can see that 0.5 is indeed the best threshold giving us 80% accuracy
# But is accuracy the best metric for this dataset? 
# No. It is not. We need to check the Recall and Precision to know whther our model is actually good or not
scores


# In[63]:


plt.plot(thresholds, scores)


# In[64]:


df_pred = pd.DataFrame()

df_pred['probability'] = y_pred
df_pred['prediction'] = churn_decision.astype('int')
df_pred['actual'] = y_val


# In[65]:


df_pred['correct'] = df_pred.prediction == df_pred.actual


# In[66]:


df_pred.correct.mean()


# # Model Interpretation

# dict(zip(dv.get_feature_names_out(), model.coef_[0].round(3)))

# In[67]:


small = ['contract','tenure','monthlycharges']


# In[68]:


df_train[small].iloc[:10].to_dict(orient='records')


# In[69]:


dicts_train_small = df_train[small].to_dict(orient='records')
dicts_val_small = df_val[small].to_dict(orient='records')


# In[70]:


dv_small = DictVectorizer(sparse = False)
dv_small.fit(dicts_train_small)


# In[71]:


X_train_small = dv_small.transform(dicts_train_small)


# In[72]:


model_small = LogisticRegression()
model_small.fit(X_train_small, y_train)


# In[73]:


w0 = model_small.intercept_[0]


# In[74]:


w0


# In[75]:


w = model_small.coef_[0]

w.round(3)


# In[76]:


dict(zip(dv_small.get_feature_names_out(), w.round(3)))


# In[77]:


sigmoid(w0 - 0.948 + 50 * 0.027 + 5 * (-0.036))


# In[78]:


-2.47 - 0.948 + 50 * 0.027 + 5 * (-0.036)


# In[79]:


sigmoid(_)


# # Using the model

# In[80]:


dicts_full_train = df_full_train[categorical + numerical].to_dict(orient='records')


# In[81]:


dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)


# In[82]:


y_full_train = df_full_train.churn.values


# In[83]:


model = LogisticRegression()
model.fit(X_full_train, y_full_train)


# In[84]:


dicts_test = df_test[categorical + numerical].to_dict(orient='records')


# In[85]:


X_test = dv.transform(dicts_test)


# In[86]:


y_pred = model.predict_proba(X_test)[:,1]


# In[87]:


#churn_decision = (y_pred >= 0.5)


# In[88]:


(churn_decision == y_test).mean()


# # Testing the model

# In[152]:


customer = dicts_test[-1]


# In[153]:


customer


# In[90]:


X_small = dv.transform([customer])


# In[91]:


model.predict_proba(X_small)[0,1]


# In[92]:


y_test[-1]


# # Accuracy

# In[93]:


len(y_val)


# In[94]:


(y_val == churn_decision).sum()


# In[95]:


#As we see our model is ~80% accurate but this is not a the best metric to check especially for churn prediction

1126/1409


# # Confusuin Matrix

# In[96]:


# A confusion matrix tells us how many
#TP = True Positives
#TN = True Negatives
#FP = False Positives
#FN = False Negatives

#our model have predicted

model.predict(X_train) #Hard Predictions


y_pred = model.predict_proba(X_val)[:, 1] #Soft Predictions


# In[97]:


actual_positive = (y_val == 1)
actual_negative = (y_val == 0)


# In[98]:


t = 0.5
predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)


# In[99]:


tp = (predict_positive & actual_positive).sum() # 215
tn = (predict_negative & actual_negative).sum() # 921

fp = (predict_positive & actual_negative).sum() 
fn = (predict_negative & actual_positive).sum() 


# In[100]:


#This tells us that:
# TN = 921, which means our model correctly predicted 921 non churners
# TP - 215, which means our model correctly predicted 215 churners

#But here is where it gets interesting
# FP = 102, which means our model predicted 102 customers as churners but they were actually non churners
# FN = 171, which means our model predicted 171 customers as non churners but they were actually churners


#This tells us that eve with an accuarcy of 80% our model is failing to predict 44% of churners
print(tp, tn, fp, fn)


# In[101]:


confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])

confusion_matrix


# In[102]:


(confusion_matrix / confusion_matrix.sum()).round(2)


# # Precision and Recall

# In[103]:


accuracy = (tp + tn) / (tp + tn + fp + fn)
accuracy


# ## The easiest way to remember
# 
# Think about a metal detector looking for gold:
# 
# - Precision: "Of everything I said was gold, how much actually was gold?"
# 
# - Recall: "Of all the gold that was actually there, how much did I find?"

# ### Precision

# In[104]:


#Precision tells us out of all the people who actually churned how many did we correctly identify

precision = tp / (tp + fp) #311
precision

# We were going to give churn mails to 311 (tp + fp) people but only 210 (tp) were actually going to churn
# So a 32.1% error rate


# In[105]:


precision_error = 1 - precision
precision_error.round(2)


# ### Recall

# In[106]:


#Recall tells us out of all the churners we predicted how many times is the model correct

recall = tp / (tp + fn)


# In[107]:


recall


# In[108]:


recall_error = 1 - recall

recall_error.round(2)

# This means that we failed to identify 44% of actual positive 


# # ROC Curve

# ## TPR and FPR

# In[109]:


tpr = tp / (tp + fn)
tpr

#TPR and Recall are same


# In[110]:


fpr = fp / (fp + tn)
fpr


# In[111]:


scores = []
threshold = np.linspace(0,1,101)

for t in threshold:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)

    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positive & actual_negative).sum() 
    fn = (predict_negative & actual_positive).sum() 

    scores.append((float(t),float(tp),float(fp),float(fn),float(tn)))


# In[112]:


columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
df_scores = pd.DataFrame(scores, columns= columns)


# In[113]:


#Different confusion matrices at different thresholds

#We should always aim for a lower FPR and a higher TPR
# The ideal FPR and TPR point is 0 and 1 which is called a 'North Star' in the ROC Curve
df_scores[::10]


# In[114]:


df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)


# In[115]:


df_scores[::10]


#This is the ROC Curve for our predictions
plt.plot(df_scores.threshold, df_scores['tpr'], label = 'TPR')
plt.plot(df_scores.threshold, df_scores['fpr'], label = 'FPR')
plt.legend()


# ## Random Model

# In[116]:


np.random.seed(1)
y_rand = np.random.uniform(0,1,size = len(y_val))


# In[117]:


((y_rand >= 0.5) == y_val).mean()


# In[118]:


#Creating a Random Model with random predictions to compare it with our model

def tpr_fpr_df(y_val, y_pred):
    scores = []
    threshold = np.linspace(0,1,101)

    for t in threshold:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()

        fp = (predict_positive & actual_negative).sum() 
        fn = (predict_negative & actual_positive).sum() 

        scores.append((float(t),float(tp),float(fp),float(fn),float(tn)))

    columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
    df_scores = pd.DataFrame(scores, columns= columns)

    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

    return df_scores


# In[119]:


df_rand = tpr_fpr_df(y_val, y_rand)


# In[120]:


df_rand[::10]


# In[121]:


plt.plot(df_rand.threshold, df_rand['tpr'], label = 'TPR')
plt.plot(df_rand.threshold, df_rand['fpr'], label = 'FPR')
plt.legend()


# ## Ideal Model

# In[122]:


#Lastly, creating an Ideal model which at a threshold of 0.78 creates a North Star
num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()

num_neg, num_pos


# In[123]:


y_ideal = np.repeat([0,1], [num_neg, num_pos])
y_ideal


# In[124]:


y_ideal_pred = np.linspace(0,1,len(y_val))


# In[125]:


((y_ideal_pred >= 0.726) == y_ideal).mean()


# In[126]:


df_ideal = tpr_fpr_df(y_ideal, y_ideal_pred)


# In[127]:


plt.plot(df_ideal.threshold, df_ideal['tpr'], label = 'TPR')
plt.plot(df_ideal.threshold, df_ideal['fpr'], label = 'FPR')
plt.legend()


# ## Putting everything together

# In[128]:


plt.plot(df_scores.threshold, df_scores['tpr'], label = 'TPR')
plt.plot(df_scores.threshold, df_scores['fpr'], label = 'FPR')

# plt.plot(df_rand.threshold, df_rand['tpr'], label = 'TPR')
# plt.plot(df_rand.threshold, df_rand['fpr'], label = 'FPR')

plt.plot(df_ideal.threshold, df_ideal['tpr'], label = 'TPR', color = 'black')
plt.plot(df_ideal.threshold, df_ideal['fpr'], label = 'FPR', color = 'black')

plt.legend()


# In[129]:


plt.figure(figsize=(5,5))

plt.plot(df_scores.fpr, df_scores.tpr, label='model')
plt.plot([0,1], [0,1], label='random')

# plt.plot(df_rand.fpr, df_rand.tpr, label='random')
plt.plot(df_ideal.fpr, df_ideal.tpr, label='ideal')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()


# In[130]:


fpr, tpr, threshold = roc_curve(y_val, y_pred)


# In[131]:


plt.figure(figsize=(5,5))

plt.plot(fpr, tpr, label='model')
plt.plot([0,1], [0,1], label='random')

# plt.plot(df_rand.fpr, df_rand.tpr, label='random')
# plt.plot(df_ideal.fpr, df_ideal.tpr, label='ideal')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()


# In[ ]:





# In[132]:


auc(df_scores.fpr, df_scores.tpr)


# In[133]:


auc(fpr, tpr)


# In[ ]:





# In[134]:


roc_auc_score(y_val, y_pred)


# In[135]:


import random


# In[136]:


neg = y_pred[y_val == 0]
pos = y_pred[y_val == 1]


# In[137]:


n = 1000
success = 0

for i in range(n):
    pos_ind = random.randint(0, len(pos) - 1)
    neg_ind = random.randint(0, len(neg) - 1)

    if pos[pos_ind] > neg[neg_ind]:
        success = success + 1

success / n


# # Cross Validation

# In[138]:


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv= DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter = 1000)
    model.fit(X_train, y_train)

    return dv, model


# In[139]:


dv, model = train(df_train, y_train, C=0.001)


# In[140]:


def predict(df_train, dv, model):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:,1]

    return y_pred


# In[141]:


y_pred = predict(df_val, dv, model)


# In[ ]:





# In[142]:


kfold = KFold(n_splits = 10, shuffle=True, random_state=1)


# In[143]:


from tqdm.auto import tqdm


# In[144]:


n_splits = 5
for C in [0, 0.001, 0.01, 0.1, 0.5, 1, 5, 10]:
    scores = []
    kfold = KFold(n_splits = n_splits, shuffle=True, random_state=1)
    for train_idx, val_idx in tqdm(kfold.split(df_full_train)):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train(df_train, y_train)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[145]:


len(train_idx), len(val_idx)


# In[146]:


len(df_full_train)


# In[147]:


get_ipython().system('pip install tqdm')


# In[148]:


print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))


# In[149]:


dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
auc


# In[ ]:




