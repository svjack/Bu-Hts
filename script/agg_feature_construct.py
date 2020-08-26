#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os


# In[2]:


train_encoded = pd.read_csv("../data/train_store_encoded_onehot.csv")


# In[3]:


train_df = pd.read_csv("../data/train.csv")
store_df = pd.read_csv("../data/store.csv")


# In[4]:


cate_df = store_df.apply(lambda x: (x["Store"], x["StoreType"] + x["Assortment"]), axis = 1).map(lambda x: x[-1]).copy().reset_index()
cate_df.columns = ["Store", "cate"]
cate_df["Store"] = cate_df["Store"] + 1


# In[5]:


def calculate_days_num(data_df, cate_df):
    import gc
    data_df["Date"] = pd.to_datetime(data_df["Date"])
    merge_df = pd.merge(data_df[["Date", "Store", "Sales"]], cate_df, on = "Store", how = "inner")
    print("merge_df shape : {}".format(merge_df.shape))
    from functools import reduce
    ordered_intersection_dates = sorted(pd.to_datetime(sorted(reduce(lambda a, b: a.intersection(b),map(lambda x: set(x.tolist()),merge_df.groupby("cate").apply(dict).map(lambda inner_dict:inner_dict["Date"]).values.tolist())))))
    ordered_intersection_dates = pd.Series(ordered_intersection_dates)
    #return ordered_intersection_dates
    sales_date_intersection = merge_df.copy()
    del merge_df
    gc.collect()
    sales_date_intersection = sales_date_intersection[sales_date_intersection["Date"].isin(ordered_intersection_dates)].copy()
    def transform_dict_to_df(row):
        Store, dict_ = row["cate"], row[0]
        Date = dict_["Date"].tolist()
        Sales = dict_["Sales"].tolist()
        df = pd.DataFrame(list(zip(*[Date, Sales])))
        df.columns = ["Date", Store]
        return df
    before_reduce_list = sales_date_intersection.groupby("cate").apply(dict).reset_index().apply(
    transform_dict_to_df
, axis = 1).values.tolist()
    #return before_reduce_list
    before_reduce_list = list(map(lambda x: x.groupby("Date").sum().reset_index(), before_reduce_list))
    sales_cate_format_df = reduce(lambda a, b: pd.merge(a, b, on = "Date", how = "inner"), before_reduce_list)
    return sales_cate_format_df


# In[6]:


sales_cate_format_df = calculate_days_num(train_df, cate_df[cate_df["cate"].isin(cate_df["cate"].value_counts()[cate_df["cate"].value_counts() > 70].index.tolist())])


# In[7]:


sales_cate_format_df["total"] = sales_cate_format_df.iloc[:, 1:].apply(lambda x: x.sum(), axis = 1)


# In[8]:


from functools import reduce
sales_cate_format_df_up = sales_cate_format_df[sales_cate_format_df.iloc[:, 1:].apply(lambda x: reduce(lambda a, b: a * b ,map(int,map(bool, x))), axis = 1) > 0]


# In[9]:


df = sales_cate_format_df_up.copy()
df.index = pd.to_datetime(df["Date"])
dates = df["Date"].copy()
del df["Date"]
df = df.asfreq("D")
df = df.interpolate(method = "linear")


# In[10]:


before_reduce_by_cate_df = pd.merge(cate_df, train_encoded, on = "Store", how = "inner")


# In[11]:


before_reduce_by_cate_df["id"] = before_reduce_by_cate_df[["cate", "Date"]].apply(lambda x: "{}_{}".format(x["cate"], x["Date"]), axis = 1)


# In[12]:


reduce_by_id = before_reduce_by_cate_df[set(before_reduce_by_cate_df.columns.tolist()).difference(set(["Store", "cate"]))].groupby("id").apply(dict)


# In[13]:


def produce_agg_measure(same_id_df, agg_funcs = {"max":np.max, "min":np.min, "count":len, "mean":np.mean}):
    if "id" in same_id_df.columns.tolist():
        del same_id_df["id"]
    same_id_df["Date"] = pd.to_datetime(same_id_df["Date"]).map(lambda x: (x - pd.to_datetime("1970-01-01")).days)
    agg_series_dict = dict(map(lambda t2: (t2[0] ,same_id_df.apply(t2[-1], axis = 0)), agg_funcs.items()))
    def rename_index(s, agg_name):
        s.index = list(map(lambda index: "{}_{}".format(index, agg_name) ,s.index.tolist()))
        return s
    agg_series_dict = dict(map(lambda t2: (t2[0] ,rename_index(t2[1], t2[0])), agg_series_dict.items()))
    return pd.concat(list(agg_series_dict.values()), axis = 0)


# In[14]:


data_part = pd.concat(reduce_by_id.map(lambda dict_: produce_agg_measure(pd.DataFrame.from_dict(dict(map(lambda t2: (t2[0], t2[1].tolist()) ,dict_.items()))))).tolist(), axis = 1)
data_part.columns = reduce_by_id.index.tolist()


# In[17]:


def retrieve_data(input_df, cate):
    req_part = input_df[filter(lambda col: col.startswith(cate),input_df.columns.tolist())].copy()
    req_part.columns = list(map(lambda col: col[3:], req_part.columns.tolist()))
    req_part = req_part.T
    req_part.columns = list(map(lambda col: "{}_{}".format(col, cate), req_part.columns.tolist()))
    req_part.index = pd.to_datetime(req_part.index)
    return req_part


# In[18]:


lookup_dict = dict(map(lambda col: (col, retrieve_data(data_part, col)) ,set(df.columns.tolist()).difference(set(["total"]))))


# In[20]:


def retrieve_total_part(lookup_dict):
    from functools import reduce
    colnames = list(map(lambda x: x[:-3], list(lookup_dict.values())[0].columns.tolist()))
    keys = list(lookup_dict.keys())
    cols = list(set(map(lambda x: x[:x.rfind("_")], colnames)))
    aggs = list(set(map(lambda x: x[x.rfind("_") + 1:], colnames)))
    
    vals_list = []
    for col in cols:
        for agg_name in aggs:
            req = []
            for cate_key in keys:
                s = lookup_dict[cate_key]["{}_{}_{}".format(col, agg_name, cate_key)]
                req.append(s)
            if agg_name == "max":
                val_s = pd.concat(req, axis = 1).dropna().apply(np.max, axis = 1)
            elif agg_name == "min":
                val_s = pd.concat(req, axis = 1).dropna().apply(np.min, axis = 1)
            elif agg_name == "count":
                val_s = pd.concat(req, axis = 1).dropna().apply(np.sum, axis = 1)
            else:
                val_s = pd.concat(req, axis = 1).dropna().apply(np.mean, axis = 1)
            val_s.name = "{}_{}_{}".format(col, agg_name, "total")
            vals_list.append(val_s)
    return pd.concat(vals_list, axis = 1)


# In[21]:


total_data_part = retrieve_total_part(lookup_dict)


# In[23]:


lookup_df = reduce(lambda a, b: pd.merge(a, b, left_index=True, right_index = True, how = "inner"), lookup_dict.values())


# In[25]:


total_data_part_asfreq_D = total_data_part.asfreq("D").sort_index().fillna(method = "pad")


# In[26]:


lookup_df_asfreq_D = lookup_df.asfreq("D").sort_index().fillna(method = "pad")


# In[27]:


df_add_lookup = pd.merge(df, lookup_df_asfreq_D, left_index = True, right_index =True, how = "inner")


# In[28]:


df_add_lookup = pd.merge(df_add_lookup, total_data_part_asfreq_D, left_index = True, right_index =True, how = "inner")


# In[32]:


df_add_lookup.to_csv("../data/df_add_lookup.csv", index = True)


# In[ ]:




