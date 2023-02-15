import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

news = pd.read_csv('fakenews.csv')
news.head(3)

news[news['real'] == 0].head()['title']

news.shape

news.info()

def nulls_info(df):
  c = df.isnull().sum().rename('count')
  d = df.isnull().mean().rename('distribution')*100
  nulls = pd.concat([c,d], axis =1)
  nulls.index.name = 'column_name'
  nulls.reset_index(inplace = True)
  return nulls

nulls_info(news)

### lets remove the records contating the null values

news = news[~news['news_url'].isnull()]
news = news[~news['source_domain'].isnull()]

nulls_info(news)

news['real'].value_counts(normalize = True)

"""Arriving at new columns"""

news['word_count'] = news['title'].apply(lambda x : len(x.split(' ')))

news['real'].value_counts(normalize = True).plot.bar()
plt.show()

### The distribution looks good in the sample data, 25% could be the right wieghtage for a news being fake,

rectify_sampling_imbalance = True

if rectify_sampling_imbalance == True:
  features = news.drop('real', 1)
  target = news['real']
  from imblearn.over_sampling import RandomOverSampler
  ros = RandomOverSampler(random_state=100)
  featues_ros, target_ros = ros.fit_resample(features,target)
  news_oversampled = pd.concat([featues_ros,target_ros], axis = 1)
  news = news_oversampled

X = news['title']
y = news['real']

sns.countplot(y)
plt.show()

from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test = train_test_split(X,y, train_size = 0.70, random_state = 100, stratify = y)

"""Tokenizer

Transforms each text in texts to a sequence of integers. Each item in texts can also be a list, in which case we assume each item of that list to be a token. Only top num_words-1 most frequent words will be taken into account. Only words known by the tokenizer will be taken into account.
"""

from tensorflow import keras as kr

tokn = kr.preprocessing.text.Tokenizer(num_words=None,
                                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                      lower=True,
                                      split=' ',
                                      char_level=False,
                                      oov_token=None)

tokn.fit_on_texts(X_train)

"""text to sequence


"""

X_train_tokn = tokn.texts_to_sequences(X_train)

X_test_tokn = tokn.texts_to_sequences(X_test)

max_word_count = int(news['word_count'].quantile(0.80))

X_train_padded = kr.preprocessing.sequence.pad_sequences(X_train_tokn, maxlen = max_word_count , padding = 'post')

X_test_padded = kr.preprocessing.sequence.pad_sequences(X_test_tokn, maxlen = max_word_count , padding = 'post')

X_train_padded

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_padded_sc = sc.fit_transform(X_train_padded)

X_test_padded_sc = sc.transform(X_test_padded)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(min_samples_leaf = 30)

dt.fit(X_train_padded_sc,y_train)

"""Lets check the matrix for the Decision Tree"""

def evaluate_matrics (estimator):
  from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,confusion_matrix
  
  train_pred = estimator.predict(X_train_padded_sc)
  test_pred = estimator.predict(X_test_padded_sc)

  accuracy_train = round(accuracy_score(y_train,train_pred),4)
  accuracy_test = round(accuracy_score(y_test,test_pred),4)
  accuracy_delta = round(accuracy_train-accuracy_test,4)

  precision_train = round(precision_score(y_train,train_pred),4)
  precision_test = round(precision_score(y_test,test_pred),4)
  precision_delta = round(precision_train-precision_test,4)

  recall_train = round(recall_score(y_train,train_pred),4)
  recall_test = round(recall_score(y_test,test_pred),4)
  recall_delta = round(recall_train-recall_test,4)
  
  f1_train = round(f1_score(y_train,train_pred),4)
  f1_test = round(f1_score(y_test,test_pred),4)
  f1_delta = round(f1_train-f1_test,4)

  roc_auc_train = round(roc_auc_score(y_train,train_pred),4)
  roc_auc_test = round(roc_auc_score(y_test,test_pred),4)
  roc_acu_delta = round(roc_auc_train-roc_auc_test,4)

  confusion_train = pd.DataFrame(confusion_matrix(y_train,train_pred))
  specificity_train = round(confusion_train.iloc[0,0]/(confusion_train.iloc[0,0]+confusion_train.iloc[0,1]),4)
  confusion_test = pd.DataFrame(confusion_matrix(y_test,test_pred))
  specificity_test = round(confusion_test.iloc[0,0]/(confusion_test.iloc[0,0]+confusion_test.iloc[0,1]),4)
  specificity_delat = round(specificity_train - specificity_test,4)


  metrics = ['accuracy','specificity','precision','recall','f1_score','roc_auc']
  train_metric = [accuracy_train,specificity_train,precision_train,recall_train,f1_train,roc_auc_train]
  test_metric = [accuracy_test,specificity_test,precision_test,recall_test,f1_test,roc_auc_test]
  delta_metric = [accuracy_delta,specificity_delat,precision_delta,recall_delta,f1_delta,roc_acu_delta]


  metrics_df = pd.DataFrame({'metric':metrics,'train':train_metric,'test':test_metric,'delta':delta_metric})
  return metrics_df

evaluate_matrics(dt)

"""This model is highly overfitting

Need to increase the model complexity
"""

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 200, min_samples_leaf = 30, criterion ='gini')

rfc.fit(X_train_padded_sc,y_train)

evaluate_matrics(rfc)

from sklearn.model_selection import GridSearchCV

estimator = RandomForestClassifier()

params = {'n_estimators' : [100,200,300], 
          'min_samples_leaf' : [50,30,20], 
          'criterion' : ['gini']}

gridsearch = GridSearchCV(estimator, 
             param_grid = params, 
             scoring='accuracy', 
             n_jobs= -1, refit=True,
             verbose=1)

gridsearch.fit(X_train_padded_sc,y_train)

best_estimator_rf = gridsearch.best_estimator_

evaluate_matrics(best_estimator_rf)

"""Lets Try an XG Boost Model"""

from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X_train_padded_sc,y_train)

evaluate_matrics(xgb)

final_model = best_estimator_rf

"""**Exporting the files for Deployement**

1. maximum word count
2. tokenize object
3. scaler object
4. classifier
"""

import pickle

pickle.dump(max_word_count,open('word_counter.pkl','wb'))
pickle.dump(tokn,open('tokenizer.pkl','wb'))
pickle.dump(sc,open('scaler.pkl','wb'))
pickle.dump(final_model,open('classifier.pkl','wb'))

"""**End of the document**"""