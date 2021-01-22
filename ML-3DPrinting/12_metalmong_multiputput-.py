#!/usr/bin/env python
# coding: utf-8

# In[1]:


from platform import python_version
import pandas as pd
import numpy as np 
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
import xgboost as xgb
import pickle
import joblib

print(f'python {python_version()}')
print(f'pandas {pd.__version__}')
print(f'numpy {np.__version__}')


# In[2]:


data = pd.read_csv('./data.csv')


# In[3]:


data.head(5)


# In[4]:


indexName = data.columns.tolist()
indexName


# ## Data Analysis 

# In[5]:


copyData = data.copy()
x_data_list = ['Power','Speed','LayerThickness']
dummy_erase_list = ['Keyhole induced pore']


# In[6]:


x_data = copyData[x_data_list]
y_data = copyData.drop(x_data_list, axis=1) 
#평균 용융풀 너비	평균 용융풀 깊이	용융풀 안정성	결함 발생율


# ## 1. 평균 용융풀 너비

# In[7]:


y_data_o1 = y_data['o1']


# In[8]:


from sklearn.linear_model import Ridge
sgd_reg_ridge = Ridge()
scores_sgd_ridge = cross_val_score(sgd_reg_ridge, x_data, y_data_o1, cv=5,  verbose=2,scoring='r2')
Ridge_result = scores_sgd_ridge.mean()
Ridge_result


# In[9]:


from sklearn.linear_model import Lasso
sgd_reg_lasso = Lasso(random_state = 42)
scores_sgd_lasso = cross_val_score(sgd_reg_lasso, x_data, y_data_o1, cv=5, n_jobs=-1, scoring = "r2")
Lasso_result = scores_sgd_lasso.mean()
Lasso_result


# In[10]:


from sklearn.linear_model import ElasticNet
elastic_reg = ElasticNet(random_state = 42)
scores_elastic_reg = cross_val_score(elastic_reg, x_data, y_data_o1, cv=5, n_jobs=-1, scoring = "r2")
ElasticNet_result = scores_elastic_reg.mean()
ElasticNet_result


# In[11]:


from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(random_state = 42)
RFR.fit(x_data,y_data_o1)
scores_RFR = cross_val_score(RFR, x_data, y_data_o1, cv=2,  verbose=10,scoring='r2')
RFR_result = scores_RFR.mean()
RFR_result


# In[12]:


from xgboost import plot_importance
import xgboost
best_xgb_model = xgboost.XGBRegressor(random_state = 42)
best_xgb_model.fit(x_data,y_data_o1)
scores_xgb = cross_val_score(best_xgb_model, x_data, y_data_o1, cv=5, n_jobs=-1, scoring = "r2")
xgr_result = scores_xgb.mean()
xgr_result


# In[13]:


import lightgbm as lgb
lgbm = lgb.LGBMRegressor(random_state = 42)
lgbm.fit(x_data,y_data_o1)
scores_lgbm = cross_val_score(lgbm, x_data, y_data_o1, cv=5,  verbose=2,scoring='r2')
lgbm_result = scores_lgbm.mean()
lgbm_result


# In[14]:


joblib.dump(RFR, 'o1_pickleRFR.pkl')
joblib.dump(best_xgb_model, 'o1_picklexgb.pkl')
joblib.dump(lgbm, 'o1_picklelgbm.pkl')


# ### pickle 파일 로드 

# In[15]:


pickleFile = joblib.load('picklelgbm.pkl')


# In[16]:


ex1 = np.array([210,2200,0.04]).reshape(1,-1)


# In[17]:


resultPrediction = pickleFile.predict(ex1)
resultPrediction


# ## 2. 평균 용융풀 깊이

# In[18]:


y_data_o2 = y_data['o2']


# In[19]:


RFR = RandomForestRegressor(random_state = 42)
RFR.fit(x_data,y_data_o2)
scores_RFR = cross_val_score(RFR, x_data, y_data_o2, cv=2,  verbose=10,scoring='r2')
RFR_result = scores_RFR.mean()
RFR_result


# In[20]:


best_xgb_model = xgboost.XGBRegressor(random_state = 42)
best_xgb_model.fit(x_data,y_data_o2)
scores_xgb = cross_val_score(best_xgb_model, x_data, y_data_o2, cv=5, n_jobs=-1, scoring = "r2")
xgr_result = scores_xgb.mean()
xgr_result


# In[21]:


lgbm = lgb.LGBMRegressor(random_state = 42)
lgbm.fit(x_data,y_data_o2)
scores_lgbm = cross_val_score(lgbm, x_data, y_data_o2, cv=5,  verbose=2,scoring='r2')
lgbm_result = scores_lgbm.mean()
lgbm_result


# In[22]:


joblib.dump(RFR, 'o2_pickleRFR.pkl')
joblib.dump(best_xgb_model, 'o2_picklexgb.pkl')
joblib.dump(lgbm, 'o2_picklelgbm.pkl')


# ## 3. 용융풀 안정성

# In[23]:


y_data_o3 = y_data['o3']


# In[24]:


RFR = RandomForestRegressor(random_state = 42)
RFR.fit(x_data,y_data_o3)
scores_RFR = cross_val_score(RFR, x_data, y_data_o3, cv=2,  verbose=10,scoring='r2')
RFR_result = scores_RFR.mean()
RFR_result


# In[25]:


best_xgb_model = xgboost.XGBRegressor(random_state = 42)
best_xgb_model.fit(x_data,y_data_o3)
scores_xgb = cross_val_score(best_xgb_model, x_data, y_data_o3, cv=5, n_jobs=-1, scoring = "r2")
xgr_result = scores_xgb.mean()
xgr_result


# In[26]:


lgbm = lgb.LGBMRegressor(random_state = 42)
lgbm.fit(x_data,y_data_o3)
scores_lgbm = cross_val_score(lgbm, x_data, y_data_o3, cv=5,  verbose=2,scoring='r2')
lgbm_result = scores_lgbm.mean()
lgbm_result


# In[27]:


joblib.dump(RFR, 'o3_pickleRFR.pkl')
joblib.dump(best_xgb_model, 'o3_picklexgb.pkl')
joblib.dump(lgbm, 'o3_picklelgbm.pkl')


# ## 4. 결함 발생율

# In[28]:


y_data_o4 = y_data['o4']


# In[29]:


RFR = RandomForestRegressor(random_state = 42)
RFR.fit(x_data,y_data_o4)
scores_RFR = cross_val_score(RFR, x_data, y_data_o4, cv=2,  verbose=10,scoring='r2')
RFR_result = scores_RFR.mean()
RFR_result


# In[30]:


best_xgb_model = xgboost.XGBRegressor(random_state = 42)
best_xgb_model.fit(x_data,y_data_o4)
scores_xgb = cross_val_score(best_xgb_model, x_data, y_data_o4, cv=5, n_jobs=-1, scoring = "r2")
xgr_result = scores_xgb.mean()
xgr_result


# In[31]:


lgbm = lgb.LGBMRegressor(random_state = 42)
lgbm.fit(x_data,y_data_o4)
scores_lgbm = cross_val_score(lgbm, x_data, y_data_o4, cv=5,  verbose=2,scoring='r2')
lgbm_result = scores_lgbm.mean()
lgbm_result


# In[32]:


joblib.dump(RFR, 'o4_pickleRFR.pkl')
joblib.dump(best_xgb_model, 'o4_picklexgb.pkl')
joblib.dump(lgbm, 'o4_picklelgbm.pkl')


# In[ ]:





# In[ ]:




