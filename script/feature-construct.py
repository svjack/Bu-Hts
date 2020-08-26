#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


train = pd.read_csv("../data/train.csv", parse_dates=True, low_memory=False, index_col = "Date")
store = pd.read_csv("../data/store.csv", low_memory=False)
store.fillna(0, inplace = True)


# In[3]:


train["Year"] = train.index.year
train["Month"] = train.index.month
train["Day"] = train.index.day
train["WeekOfYear"] = train.index.weekofyear
train["SalesPerCustomer"] = train["Sales"] / train["Customers"]


# In[4]:


train = train[(train["Open"] != 0) & (train["Sales"] != 0)].copy()


# In[5]:


train_store = pd.merge(train, store, how = "inner", on = "Store")


# In[6]:


train_store_expand_date = pd.merge(train.reset_index(), store, how = "inner", on = "Store")


# In[7]:


schoolh_by_store = train_store_expand_date[["Store", "SchoolHoliday", "Date"]].groupby("Store")[["SchoolHoliday", "Date"]].apply(dict)


# In[8]:


from functools import reduce
inter_dates = sorted(list(reduce(lambda a, b: a.intersection(b) ,schoolh_by_store.map(lambda dict_: set(dict_["Date"].tolist())).tolist())))


# In[9]:


def retrieve_sd_list(dict_):
    SchoolHoliday = dict_["SchoolHoliday"].tolist()
    Date = dict_["Date"].tolist()
    assert len(SchoolHoliday) == len(Date)
    req = []
    for ele in inter_dates:
        idx = Date.index(ele)
        req.append(SchoolHoliday[idx])
    return req
schoolh_inter_idxes = schoolh_by_store.map(
    retrieve_sd_list
)


# In[10]:


school_holiday_store_df = schoolh_inter_idxes.map(lambda x: "".join(map(str,x))).reset_index()


# In[11]:


from copy import deepcopy 
store_cp = deepcopy(store)
store_cp_info = pd.merge(store_cp, school_holiday_store_df, on = "Store", how = "inner")


# In[12]:


from functools import reduce
PromoInterval_cnt_series = pd.Series(reduce(lambda a, b: a + b ,store_cp_info["PromoInterval"].map(lambda x: list(map(lambda y :"PromoInterval_{}".format(y),x.split(","))) if type(x) == type("") else []).values.tolist())).value_counts()


# In[13]:


PromoInterval_expand_columns = PromoInterval_cnt_series.sort_index().index.tolist()


# In[14]:


store_cp_info_expand_PromoInterval = deepcopy(store_cp_info)
for col in PromoInterval_expand_columns:
    store_cp_info_expand_PromoInterval[col] = 0


# In[15]:


for ridx, (idx, r) in enumerate(store_cp_info.iterrows()):
    x = r["PromoInterval"]
    set_list = list(map(lambda y :"PromoInterval_{}".format(y),x.split(","))) if type(x) == type("") else []
    for ele in set_list:
        store_cp_info_expand_PromoInterval.iloc[ridx, store_cp_info_expand_PromoInterval.columns.get_loc(ele)] = 1


# In[16]:


store_cp_info_expand_PromoInterval = store_cp_info_expand_PromoInterval.rename(columns = {
    0: "schoolholiday_str"
})


# In[17]:


def simple_cate_encode(input_series):
    idx_value_dict = dict(enumerate(input_series.value_counts().index.tolist()))
    value_idx_dict = dict(map(lambda t2: (t2[1], t2[0]), idx_value_dict.items()))
    return pd.Series(list(map(lambda x: value_idx_dict[x], input_series.values.tolist())))
set_new_columns_dict = dict(map(lambda colname: (colname, simple_cate_encode(store_cp_info_expand_PromoInterval[colname])) ,store_cp_info_expand_PromoInterval.dtypes.map(str)[store_cp_info_expand_PromoInterval.dtypes.map(str) == "object"].index.tolist()))


# In[18]:


for colname, new_col in set_new_columns_dict.items():
    store_cp_info_expand_PromoInterval["{}_encode".format(colname)] = new_col


# In[19]:


def transform_columns(left, right):
    assert "Store" in left.columns.tolist() and "Store" in right.columns.tolist()
    right_encoded_colnames = list(filter(lambda colname: colname.endswith("_encode") ,right.columns.tolist()))
    right_encoded_colnames.remove("schoolholiday_str_encode")
    print("add num : {}".format(len(right_encoded_colnames)))
    left_replace_colnames = list(map(lambda colname: colname.replace("_encode", ""), right_encoded_colnames))          
    assert len(left_replace_colnames) == len(set(left_replace_colnames).intersection(set(left.columns.tolist())))
    left_before_merge = left.copy()
    for col in left_replace_colnames:
        del left_before_merge[col]
    right_before_merge = right.copy()
    merged = pd.merge(left = left_before_merge, right = right_before_merge, on = "Store", how = "inner")
    assert left.shape[0] == merged.shape[0]
    return merged
    
right_cols = store_cp_info_expand_PromoInterval.columns.tolist()[store_cp_info_expand_PromoInterval.columns.tolist().index("PromoInterval_Apr"):]
right_cols = ["Store"] + right_cols
train_store_encoded = transform_columns(left = train_store_expand_date, right = store_cp_info_expand_PromoInterval[right_cols])


# In[20]:


obj_cols = train_store_encoded.dtypes.map(str)[train_store_encoded.dtypes.map(str) == "object"].index.tolist()
for col in obj_cols:
    train_store_encoded["{}_encode".format(col)] = simple_cate_encode(train_store_encoded[col])
    del train_store_encoded[col]


# In[21]:


if "Open" in train_store_encoded.columns.tolist():
    del train_store_encoded["Open"]


# In[22]:


int_cols = train_store_encoded.dtypes.map(str)[train_store_encoded.dtypes.map(str) == "int64"].index.tolist()
int_cols_stats = train_store_encoded[int_cols].apply(lambda s: len(s.value_counts()), axis = 0)
bool_cols = int_cols_stats[int_cols_stats == 2]
bool_stats = int_cols_stats.loc[bool_cols.index]
others = list(set(int_cols_stats.index.tolist()).difference(set(bool_cols.index.tolist())))
other_stats = int_cols_stats.loc[others]
encode_cols = list(filter(lambda x: x.endswith("_encode"), other_stats.index.tolist()))
encode_stats = int_cols_stats.loc[encode_cols]
others = list(set(others).difference(set(encode_stats.index.tolist())))
other_stats = int_cols_stats.loc[others]


# In[23]:


int_cols_stats_list = [bool_stats, encode_stats, other_stats]
float_cols = train_store_encoded.dtypes.map(str)[train_store_encoded.dtypes.map(str) == "float64"].index.tolist()
float_stats = train_store_encoded[float_cols].apply(lambda x: len(x.value_counts()), axis = 0)
ds_stats = train_store_encoded[train_store_encoded.dtypes.map(str)[train_store_encoded.dtypes.map(str) == "datetime64[ns]"].index.tolist()].apply(lambda x: len(x.value_counts()), axis = 0)
all_cols_stats_list = int_cols_stats_list + [float_stats, ds_stats]


# In[24]:


from functools import reduce
assert reduce(lambda a, b: a + b ,map(len ,all_cols_stats_list)) == train_store_encoded.shape[1]


# In[25]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc_part = enc.fit_transform(train_store_encoded[encode_stats.index.tolist()])


# In[26]:


req = []
for col_stats_idx in set(range(len(all_cols_stats_list))).difference(set([1])):
    req.append(train_store_encoded[all_cols_stats_list[col_stats_idx].index.tolist()])
req.append(enc_part)


# In[27]:


#list(map(lambda x: x.shape, req))


# In[28]:


train_store_encoded_onehot = pd.concat(map(lambda x: x if type(x) == type(pd.DataFrame()) else pd.DataFrame(x.toarray()) ,req), axis = 1)


# In[29]:


train_store_encoded_onehot.columns, train_store_encoded_onehot.shape


# In[30]:


train_store_encoded.to_csv("../data/train_store_encoded.csv", index = False)
train_store_expand_date.to_csv("../data/train_store_expand_date.csv", index = False)
train_store_encoded_onehot.to_csv("../data/train_store_encoded_onehot.csv", index = False)


# In[ ]:




