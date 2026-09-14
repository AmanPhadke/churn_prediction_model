import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

#Parameters
print('Loading Parameters...')

C = 1.0
n_splits = 5
output_file = f'model_C={C}.bin'


# Data Preparation

print('Preparing the Data...')

data = "C:/Users/Asus/OneDrive/Desktop/Projects/Customer_Churn/churn_db.csv"
df= pd.read_csv(data)

df.columns = df.columns.str.replace(' ', '_').str.lower()

categorical_columns = list(df.select_dtypes('object').columns)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)
df.churn = (df.churn == 'yes').astype('int')


df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state=1)


numerical = ['seniorcitizen', 'tenure', 'monthlycharges', 'totalcharges']
categorical = ['gender', 'partner', 'dependents', 'phoneservice',
       'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup',
       'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies',
       'contract', 'paperlessbilling', 'paymentmethod']



#Training

print('Training the Model...')

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv= DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter = 3000)
    model.fit(X_train, y_train)

    return dv, model


def predict(df_train, dv, model):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:,1]

    return y_pred


# Validation

print('Validating the model...')
kfold = KFold(n_splits = n_splits, shuffle=True, random_state=1)
scores = []

fold = 0
for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'AUC on Fold = {fold} is {auc}')
    fold += 1

print('----------------------Validation Result------------------------')
print(f'Mean AUC Score with SD at C={C}')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# Training the final model

print('Training the final model')
dv, model = train(df_full_train, df_full_train.churn.values)
y_pred = predict(df_test, dv, model)

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)

print(f'AUC of final model = {auc}')

# Saving the model

# f_out = open(output_file, 'wb')
# pickle.dump((dv, model), f_out)
# f_out.close()

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'The model is saved as {output_file}')



