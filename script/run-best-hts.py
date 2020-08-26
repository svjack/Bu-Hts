#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from fbprophet import Prophet
import os
import json


# In[2]:


import inspect
def get_default_args(func):
    signature = inspect.signature(func)
    return{
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty 
    }

prophet_defaults = get_default_args(Prophet)


# In[3]:


hierarchy = {"total": ["aa", "ca", "ac", "cc", "dc", "da"]}


# In[4]:


df_path = "../data/df_add_lookup.csv"
df = pd.read_csv(df_path)
shift_cols = list(filter(lambda x: x.startswith("Customers") or x.startswith("Sales"),df.columns.tolist()))
for col in shift_cols:
    df[col] = df[col].shift(1)
#df = df.dropna()
df = df.fillna(0)
if "Date" in df.columns.tolist():
    df.index = pd.to_datetime(df["Date"])
    del df["Date"]
df = df.asfreq("D")


# In[5]:


head_set = set(filter(lambda x: "_" in x ,map(lambda col: col[:col.rfind("_")] ,df.columns.tolist())))
cate_set = set(filter(lambda x: "_" not in x ,map(lambda col: col[col.rfind("_") + 1:] ,df.columns.tolist())))


# In[6]:


from hts.model import FBProphetModel
from hts.model.p import *
class FBProphetModel_Add_Version(FBProphetModel):
    def create_model(self, *args, **kwargs):
        self.lookup_df = None
        has_children = bool(self.node.children)
        is_children = not has_children
        print("will create {} model {}".format("leaf" if is_children else "branch", self.node) + "-" * 10)
        if "ext_args" not in kwargs:
            return super(FBProphetModel_Add_Version, self).create_model(*args, **kwargs)
        base_kwargs = dict()
        for k, v in kwargs.items():
            if k not in ["ext_args", "add_reg"]:
                base_kwargs[k] = v
        base_model = super(FBProphetModel_Add_Version, self).create_model(*args, **base_kwargs)
        #base_model = super().create_model(*args, **kwargs)
        ext_args_dict = kwargs["ext_args"]
        if "seasonality" in ext_args_dict:
            assert type(ext_args_dict["seasonality"]) == type([])
            for seasonality_dict in ext_args_dict["seasonality"]:
                name, period, fourier_order, prior_scale = seasonality_dict["name"], seasonality_dict["period"], seasonality_dict["fourier_order"], seasonality_dict["prior_scale"]
                base_model.add_seasonality(name = name, period = period, fourier_order = fourier_order, prior_scale =prior_scale)
                print("seasonality {} {} {} {} added".format(name, period, fourier_order, prior_scale))
        if "add_reg" in kwargs and "data" in kwargs["add_reg"] and kwargs["add_reg"]["data"] is not None:
            assert "reg" in ext_args_dict
            print("reg")
            reg_prior_dict = ext_args_dict["reg"]
            self.lookup_df = kwargs["add_reg"]["data"]
            cate = self.node.key
            assert cate in cate_set
            for col in map(lambda x: "{}_{}".format(x, cate), head_set):
                if "_mean_" in col:
                    prior_scale = reg_prior_dict["{}_prior".format(col)]
                    base_model.add_regressor(col, prior_scale = prior_scale)
                    #print("reg {} added".format(col))
        return base_model
    def fit(self, **fit_args) -> 'TimeSeriesModel':
        df = self._reformat(self.node.item)

        if self.lookup_df is not None:
            lookup_df = self.lookup_df
            before_shape = df.shape
            if "Date" not in lookup_df.columns.tolist():
                lookup_df = lookup_df.reset_index()
            ds_col = list(map(lambda tt2: tt2[0] ,filter(lambda t2: t2[-1] ,lookup_df.dtypes.map(str).map(lambda x: x == "datetime64[ns]").to_dict().items())))[0]
            lookup_df = lookup_df.rename(columns = {ds_col: "ds"})
            df = pd.merge(df, lookup_df, on = "ds", how = "inner")
            print("fit merge shape {} before shape {}".format(df.shape, before_shape))
            
        with suppress_stdout_stderr():
            self.model = self.model.fit(df)
            self.model.stan_backend = None
        return self

    def predict(self,
                node: HierarchyTree,
                freq: str = 'D',
                steps_ahead: int = 1):

        df = self._reformat(node.item)
        future = self.model.make_future_dataframe(periods=steps_ahead,
                                                  freq=freq,
                                                  include_history=True)
        if self.cap:
            future['cap'] = self.cap
        if self.floor:
            future['floor'] = self.floor
        
        if self.lookup_df is not None:
            lookup_df = self.lookup_df
            before_shape = future.shape
            if "Date" not in lookup_df.columns.tolist():
                lookup_df = lookup_df.reset_index()
            ds_col = list(map(lambda tt2: tt2[0] ,filter(lambda t2: t2[-1] ,lookup_df.dtypes.map(str).map(lambda x: x == "datetime64[ns]").to_dict().items())))[0]
            lookup_df = lookup_df.rename(columns = {ds_col: "ds"})
            future = pd.merge(future, lookup_df, on = "ds", how = "inner")
            print("pred merge shape {} before shape {}".format(future.shape, before_shape))
        
        self.forecast = self.model.predict(future)
        merged = pandas.merge(df, self.forecast, on='ds')
        self.residual = (merged['yhat'] - merged['y']).values
        self.mse = numpy.mean(numpy.array(self.residual) ** 2)
        if self.cap is not None:
            self.forecast.yhat = numpy.exp(self.forecast.yhat)
        if self.transform:
            self.forecast.yhat = self.transformer.inverse_transform(self.forecast.yhat)
            self.forecast.trend = self.transformer.inverse_transform(self.forecast.trend)
            for component in ["seasonal", "daily", "weekly", "yearly", "holidays"]:
                if component in self.forecast.columns.tolist():
                    inv_transf = self.transformer.inverse_transform(getattr(self.forecast, component))
                    setattr(self.forecast, component, inv_transf)
        return self


# In[7]:


from hts.revision import *
class RevisionMethod_BU(RevisionMethod):
    def _y_hat_matrix(self, forecasts):
        print("forecasts first key: {}".format(list(forecasts.keys())[0]))
        n_cols = len(list(forecasts.keys())) + 1
        keys = range(n_cols - self.sum_mat.shape[1] - 1, n_cols - 1)
        #named_keys = list(map(lambda k_idx: list(forecasts.keys())[k_idx], keys))
        named_keys = list(set(forecasts.keys()).difference(set(["total"])))
        return y_hat_matrix(forecasts, keys=named_keys)


# In[8]:


from hts.core.regressor import *
class HTSRegressor_Add_Verison(HTSRegressor):
    def _init_revision(self):
        self.revision_method = RevisionMethod_BU(sum_mat=self.sum_mat, transformer=self.transform, name=self.method)
    def _set_model_instance(self):
        from hts import model as hts_models
        from copy import deepcopy
        model_mapping = deepcopy(hts_models.MODEL_MAPPING)
        model_mapping["prophet"] = FBProphetModel_Add_Version
        try:
            self.model_instance = model_mapping[self.model]
        except KeyError:
            raise InvalidArgumentException(f'Model {self.model} not valid. Pick one of: {" ".join(Model.names())}')
            
    def _revise(self, steps_ahead=1):
        logger.info(f'Reconciling forecasts using {self.revision_method}')
        revised = self.revision_method.revise(
            forecasts=self.hts_result.forecasts,
            mse=self.hts_result.errors,
            nodes=self.nodes
        )

        revised_columns = list(make_iterable(self.nodes))
        revised_index = self._get_predict_index(steps_ahead=steps_ahead)
        return pandas.DataFrame(revised,
                                index=revised_index,
                                columns=revised_columns)


# In[9]:


import pickle as pkl
with open("../working/serlize_m_dict.pkl", "rb") as f:
    serlize_m_dict = pkl.load(f)


# In[12]:


best_key = min(serlize_m_dict.keys())
best_kw = serlize_m_dict[best_key][-1]


# In[13]:


best_key


# In[14]:


def retrieve_clf(kw):
    from copy import deepcopy
    from time import time
    from fbprophet.diagnostics import generate_cutoffs, performance_metrics
    start_time = time()
    assert type(kw) == type(dict())
    
    n_changepoints = int(kw["n_changepoints"])
    changepoint_range = kw["changepoint_range"]
    seasonality_prior_scale = kw["seasonality_prior_scale"]
    holidays_prior_scale = kw["holidays_prior_scale"]
    changepoint_prior_scale = kw["changepoint_prior_scale"]
    
    yearly_fourier_order = int(kw["yearly_fourier_order"])
    weekly_fourier_order = int(kw["weekly_fourier_order"])
    monthly_fourier_order = int(kw["monthly_fourier_order"])

    yearly_prior = kw["yearly_prior"]
    weekly_prior = kw["weekly_prior"]
    monthly_prior = kw["monthly_prior"]

    kwargs = dict()
    kwargs["yearly_seasonality"] = False
    kwargs["daily_seasonality"] = False
    kwargs["weekly_seasonality"] = False
    kwargs["ext_args"] = {
    "seasonality": [],
    "reg": {}
}

    kwargs["ext_args"]["seasonality"].append(
    {
        "name": "yearly",
        "period": 365.25,
        "fourier_order": yearly_fourier_order,
        "prior_scale": yearly_prior
    }
)
    kwargs["ext_args"]["seasonality"].append(
    {
        "name": "weekly",
        "period": 7,
        "fourier_order": weekly_fourier_order,
        "prior_scale": weekly_prior
    }
)
    kwargs["ext_args"]["seasonality"].append(
    {
        "name": "monthly",
        "period": 30.5,
        "fourier_order": monthly_fourier_order,
        "prior_scale": monthly_prior
    }
)
    
    kwargs["n_changepoints"] = n_changepoints
    kwargs["seasonality_prior_scale"] = seasonality_prior_scale
    kwargs["holidays_prior_scale"] = holidays_prior_scale
    kwargs["changepoint_prior_scale"] = changepoint_prior_scale
    
    kwargs["changepoint_range"] = changepoint_range
    kwargs["add_reg"] = {
        "data": None, 
    }
    
    horizon = "7 days"
    period = "7 days"
    initial = "870 days"
    
    horizon, period, initial = map(pd.Timedelta, [horizon, period, initial])
    df = pd.read_csv(df_path)
    shift_cols = list(filter(lambda x: x.startswith("Customers") or x.startswith("Sales"),df.columns.tolist()))
    for col in shift_cols:
        df[col] = df[col].shift(1).copy()
    df = df.fillna(0)
    df["Date"] = pd.to_datetime(df["Date"])
    
    kwargs["add_reg"]["data"] = df
    
    other_cols = list(filter(lambda col: col not in ["Date"] + list(hierarchy.keys()) + list(hierarchy.values())[0] ,df.columns.tolist()))
    for col in other_cols:
        kwargs["ext_args"]["reg"]["{}_prior".format(col)] = kw["{}_prior".format(col)]

    cutoffs = generate_cutoffs(df = df.reset_index().rename(columns ={"Date": "ds", "total": "y"})[["ds", "y"]],horizon=horizon, period=period, initial=initial)

    cutoffs_list = list(cutoffs)

    def cutoff_forecast_gen(input_df, cutoffs_list):
        if "Date" not in input_df.columns.tolist() and "ds" not in input_df.columns.tolist():
            input_df = input_df.reset_index() 
            ds_col = "Date"
        else:
            ds_col = "Date" if "Date" in input_df.columns.tolist() else "ds"
        assert ds_col in input_df.columns.tolist()
        input_df[ds_col] = pd.to_datetime(input_df[ds_col])
        for i in range(len(sorted(cutoffs_list)[:-1])):
            time = sorted(cutoffs_list)[i]
            df_for_fit = input_df[input_df[ds_col] <= time].copy()
            pred_length = (sorted(cutoffs_list)[i + 1] - time).days
            df_for_fit.index = df_for_fit[ds_col]
            del df_for_fit[ds_col]
            yield (df_for_fit, pred_length)

    cutoffs_gen = cutoff_forecast_gen(df, cutoffs_list=cutoffs_list)

    cutoffs_gen_list = list(cutoffs_gen)

    def single_fit_predict(total_df ,clf_for_cp, df_for_fit, hierarchy, pred_length, measure_cols = "total", return_errors = True):
        assert type(measure_cols) in [type([]), type("")]
        measure_cols_list = measure_cols if type(measure_cols) == type([]) else [measure_cols]
        from copy import deepcopy
        clf = deepcopy(clf_for_cp)
        df_for_fit = df_for_fit.asfreq("D")
        model = clf.fit(df_for_fit, hierarchy)
        preds = model.predict(steps_ahead=pred_length)
        if "Date" not in total_df.columns.tolist():
            total_df = total_df.reset_index()
        if "Date" not in preds.columns.tolist():
            preds = preds.reset_index()
        
        total_ds_col = list(map(lambda tt2: tt2[0] ,filter(lambda t2: t2[-1] ,total_df.dtypes.map(str).map(lambda x: x == "datetime64[ns]").to_dict().items())))[0]
        preds_ds_col = list(map(lambda tt2: tt2[0] ,filter(lambda t2: t2[-1] ,preds.dtypes.map(str).map(lambda x: x == "datetime64[ns]").to_dict().items())))[0]
        preds_lookup_part = preds.sort_values(by = preds_ds_col, ascending = True).tail(pred_length).copy()
        total_lookup_part = total_df[total_df[total_ds_col].isin(preds_lookup_part[preds_ds_col])]
        print("total_part {} preds_part {}".format(total_lookup_part.shape, preds_lookup_part.shape))
        assert len(set(measure_cols_list).intersection(set(total_lookup_part.columns.tolist()))) == len(measure_cols_list)
        assert len(set(measure_cols_list).intersection(set(preds_lookup_part.columns.tolist()))) == len(measure_cols_list)
        preds_lookup_part = preds_lookup_part[preds_lookup_part[preds_ds_col].isin(total_lookup_part[total_ds_col])]
        total_lookup_part = total_lookup_part.sort_values(by = total_ds_col, ascending = True)
        preds_lookup_part = preds_lookup_part.sort_values(by = preds_ds_col, ascending = True)
        if not return_errors:
            return (total_lookup_part[measure_cols_list], preds_lookup_part[measure_cols_list])
        
        measure_cols_errors = (total_lookup_part[measure_cols_list] - preds_lookup_part[measure_cols_list]).applymap(abs).mean(axis = 0)
        mean_errors = measure_cols_errors.mean()
        return (measure_cols_errors, mean_errors)
    
    clf = HTSRegressor_Add_Verison(model='prophet', revision_method='BU', n_jobs=4, low_memory=False, **kwargs)
    return clf


# In[15]:


best_clf = retrieve_clf(best_kw)


# In[17]:


model = best_clf.fit(df.head(len(df) - 6 * 7), hierarchy)


# In[19]:


preds = model.predict(steps_ahead=6 * 7)


# In[20]:


import seaborn as sns
dates = pd.Series(df.index.tolist())
ax = sns.lineplot(data = preds.reset_index()[["index", "total"]][preds.reset_index()["index"].isin(dates)][preds.reset_index()["index"] > "2015-01-01"], x = "index", y = "total")
ax.axes.set_ylim(0, 1.5 *int(1e7))


# In[21]:


ax = sns.lineplot(data = df.reset_index()[df.reset_index()["Date"] > "2015-01-01"], x = "Date", y = "total")
ax.axes.set_ylim(0, 1.5 *int(1e7))


# In[22]:


comp_df = pd.merge(preds[["total"]].tail(6 * 7), df[df.index.isin(preds["total"].index.tolist())][["total"]], left_index=True, right_index=True, how = "inner")
comp_df.columns = ["preds",  "ori"]


# In[23]:


preds_df = comp_df[["preds"]].copy()
ori_df = comp_df[["ori"]].copy()
preds_df["cate"], ori_df["cate"] = "preds", "ori"
preds_df = preds_df.rename(columns={"preds": "val"})
ori_df = ori_df.rename(columns={"ori": "val"})
sns.lineplot(data = pd.concat([ori_df, preds_df], axis = 0).reset_index(), x = "index", y = "val", hue = "cate")


# In[24]:


def object_func(kw):
    from copy import deepcopy
    from time import time
    from fbprophet.diagnostics import generate_cutoffs, performance_metrics
    start_time = time()
    assert type(kw) == type(dict())
    
    n_changepoints = int(kw["n_changepoints"])
    changepoint_range = kw["changepoint_range"]
    seasonality_prior_scale = kw["seasonality_prior_scale"]
    holidays_prior_scale = kw["holidays_prior_scale"]
    changepoint_prior_scale = kw["changepoint_prior_scale"]
    
    yearly_fourier_order = int(kw["yearly_fourier_order"])
    weekly_fourier_order = int(kw["weekly_fourier_order"])
    monthly_fourier_order = int(kw["monthly_fourier_order"])

    yearly_prior = kw["yearly_prior"]
    weekly_prior = kw["weekly_prior"]
    monthly_prior = kw["monthly_prior"]

    kwargs = dict()
    kwargs["yearly_seasonality"] = False
    kwargs["daily_seasonality"] = False
    kwargs["weekly_seasonality"] = False
    kwargs["ext_args"] = {
    "seasonality": [],
    "reg": {}
}

    kwargs["ext_args"]["seasonality"].append(
    {
        "name": "yearly",
        "period": 365.25,
        "fourier_order": yearly_fourier_order,
        "prior_scale": yearly_prior
    }
)
    kwargs["ext_args"]["seasonality"].append(
    {
        "name": "weekly",
        "period": 7,
        "fourier_order": weekly_fourier_order,
        "prior_scale": weekly_prior
    }
)
    kwargs["ext_args"]["seasonality"].append(
    {
        "name": "monthly",
        "period": 30.5,
        "fourier_order": monthly_fourier_order,
        "prior_scale": monthly_prior
    }
)
    
    kwargs["n_changepoints"] = n_changepoints
    kwargs["seasonality_prior_scale"] = seasonality_prior_scale
    kwargs["holidays_prior_scale"] = holidays_prior_scale
    kwargs["changepoint_prior_scale"] = changepoint_prior_scale
    
    kwargs["changepoint_range"] = changepoint_range
    kwargs["add_reg"] = {
        "data": None, 
    }
    
    horizon = "7 days"
    period = "7 days"
    initial = "870 days"
    
    horizon, period, initial = map(pd.Timedelta, [horizon, period, initial])
    df = pd.read_csv(df_path)
    shift_cols = list(filter(lambda x: x.startswith("Customers") or x.startswith("Sales"),df.columns.tolist()))

    for col in shift_cols:
        df[col] = df[col].shift(1).copy()

    df = df.fillna(0)
    df["Date"] = pd.to_datetime(df["Date"])
    
    kwargs["add_reg"]["data"] = df
    
    other_cols = list(filter(lambda col: col not in ["Date"] + list(hierarchy.keys()) + list(hierarchy.values())[0] ,df.columns.tolist()))
    for col in other_cols:
        kwargs["ext_args"]["reg"]["{}_prior".format(col)] = kw["{}_prior".format(col)]

    cutoffs = generate_cutoffs(df = df.reset_index().rename(columns ={"Date": "ds", "total": "y"})[["ds", "y"]],horizon=horizon, period=period, initial=initial)

    cutoffs_list = list(cutoffs)

    def cutoff_forecast_gen(input_df, cutoffs_list):
        if "Date" not in input_df.columns.tolist() and "ds" not in input_df.columns.tolist():
            input_df = input_df.reset_index() 
            ds_col = "Date"
        else:
            ds_col = "Date" if "Date" in input_df.columns.tolist() else "ds"
        assert ds_col in input_df.columns.tolist()
        input_df[ds_col] = pd.to_datetime(input_df[ds_col])
        for i in range(len(sorted(cutoffs_list)[:-1])):
            time = sorted(cutoffs_list)[i]
            df_for_fit = input_df[input_df[ds_col] <= time].copy()
            pred_length = (sorted(cutoffs_list)[i + 1] - time).days
            df_for_fit.index = df_for_fit[ds_col]
            del df_for_fit[ds_col]
            yield (df_for_fit, pred_length)

    cutoffs_gen = cutoff_forecast_gen(df, cutoffs_list=cutoffs_list)

    cutoffs_gen_list = list(cutoffs_gen)

    def single_fit_predict(total_df ,clf_for_cp, df_for_fit, hierarchy, pred_length, measure_cols = "total", return_errors = True):
        assert type(measure_cols) in [type([]), type("")]
        measure_cols_list = measure_cols if type(measure_cols) == type([]) else [measure_cols]
        from copy import deepcopy
        clf = deepcopy(clf_for_cp)
        df_for_fit = df_for_fit.asfreq("D")
        model = clf.fit(df_for_fit, hierarchy)
        preds = model.predict(steps_ahead=pred_length)
        if "Date" not in total_df.columns.tolist():
            total_df = total_df.reset_index()
        if "Date" not in preds.columns.tolist():
            preds = preds.reset_index()
        
        total_ds_col = list(map(lambda tt2: tt2[0] ,filter(lambda t2: t2[-1] ,total_df.dtypes.map(str).map(lambda x: x == "datetime64[ns]").to_dict().items())))[0]
        preds_ds_col = list(map(lambda tt2: tt2[0] ,filter(lambda t2: t2[-1] ,preds.dtypes.map(str).map(lambda x: x == "datetime64[ns]").to_dict().items())))[0]
        preds_lookup_part = preds.sort_values(by = preds_ds_col, ascending = True).tail(pred_length).copy()
        total_lookup_part = total_df[total_df[total_ds_col].isin(preds_lookup_part[preds_ds_col])]
        print("total_part {} preds_part {}".format(total_lookup_part.shape, preds_lookup_part.shape))
        assert len(set(measure_cols_list).intersection(set(total_lookup_part.columns.tolist()))) == len(measure_cols_list)
        assert len(set(measure_cols_list).intersection(set(preds_lookup_part.columns.tolist()))) == len(measure_cols_list)
        preds_lookup_part = preds_lookup_part[preds_lookup_part[preds_ds_col].isin(total_lookup_part[total_ds_col])]
        total_lookup_part = total_lookup_part.sort_values(by = total_ds_col, ascending = True)
        preds_lookup_part = preds_lookup_part.sort_values(by = preds_ds_col, ascending = True)
        if not return_errors:
            return (total_lookup_part[measure_cols_list], preds_lookup_part[measure_cols_list])
        
        measure_cols_errors = (total_lookup_part[measure_cols_list] - preds_lookup_part[measure_cols_list]).applymap(abs).mean(axis = 0)
        mean_errors = measure_cols_errors.mean()
        return (measure_cols_errors, mean_errors)
    
    clf = HTSRegressor_Add_Verison(model='prophet', revision_method='BU', n_jobs=4, low_memory=False, **kwargs)
    
    def construct_df_cv_format(ori_df, pred_df, max_date, add_one = True):
        assert ori_df.columns.tolist() == pred_df.columns.tolist() and ori_df.shape == pred_df.shape
        ori_series_list = []
        pred_series_list = []
        cols = ori_df.columns.tolist()
        for col in cols:
            ori_series_list.append(ori_df[col])
            pred_series_list.append(pred_df[col])
        
        def bind_single_series(ori_s, pred_s):
            bind_df = pd.concat(list(map(pd.Series,[ori_s.values.tolist(), pred_s.values.tolist()])), axis = 1)
            assert bind_df.shape[0] == len(ori_s)
            bind_df.columns = ["y", "yhat"]
            if add_one:
                bind_df["y"] += 1
                bind_df["yhat"] += 1
            bind_df["ds"] = pd.date_range(max_date - pd.Timedelta("{}d".format(len(ori_s) - 1)), max_date)
            bind_df["cutoff"] = max_date - pd.Timedelta("{}d".format(len(ori_s)))
            return bind_df

        before_performance_list = list(map(lambda idx: ( ori_series_list[idx].name,bind_single_series(ori_series_list[idx], pred_series_list[idx])), range(len(ori_series_list))))
        different_part_dict = dict(map(lambda t2: (t2[0] ,performance_metrics(t2[-1], ["mape"])), before_performance_list))
        return dict(map(lambda t2: (t2[0], t2[1]["mape"].mean()) ,different_part_dict.items()))

    use_mape = True
    if use_mape:
        cutoffs_run_conclusions = list(map(lambda df_and_step:  (pd.Series(df_and_step[0].index.tolist()).max() ,single_fit_predict(df[["Date"] + list(hierarchy.keys()) + list(hierarchy.values())[0]] ,clf, df_and_step[0], hierarchy, df_and_step[1], ["total"], return_errors = False)), cutoffs_gen_list))
        dict_list = list(map(lambda t2: construct_df_cv_format(t2[-1][0], t2[-1][1], t2[0]), cutoffs_run_conclusions))
        from collections import defaultdict
        req = defaultdict(list)
        for dict_ in dict_list:
            for k, v in dict_.items():
                req[k].append(v)
        req_cp = dict()
        for k, v in req.items():
            req_cp[k] = pd.Series(v).mean()
        score = pd.Series(list(req_cp.values())).mean()
    else:
        cutoffs_run_conclusions = list(map(lambda df_and_step:  (pd.Series(df_and_step[0].index.tolist()).max() ,single_fit_predict(df ,clf, df_and_step[0], hierarchy, df_and_step[1], ["total"], return_errors = True)), cutoffs_gen_list))
            
        cutoffs_run_conclusions_df = pd.DataFrame(cutoffs_run_conclusions)
        cutoffs_run_conclusions_df.columns = ["Date", "error"]
        data_unzipped = pd.concat(list(map(pd.Series ,cutoffs_run_conclusions_df["error"].map(lambda x: dict(x[0])).values.tolist())), axis = 1).T
        cutoffs_run_conclusions_df[data_unzipped.columns.tolist()] = data_unzipped
        cutoffs_run_conclusions_df = cutoffs_run_conclusions_df[["Date"] + data_unzipped.columns.tolist()]
        score = cutoffs_run_conclusions_df["total"].mean()
    
    '''
    import pickle as pkl
    import os
    retain_size = 100
    serlize_path = "../working/serlize_m_dict.pkl"
    if os.path.exists(serlize_path):
        with open(serlize_path, "rb") as f:
            ori_serlize_m_dict = pkl.load(f)
        if len(ori_serlize_m_dict) > retain_size:
            ori_serlize_m_dict = dict(list(sorted(ori_serlize_m_dict.items(), key = lambda t2: t2[0]))[:retain_size])
    else:
        ori_serlize_m_dict = {}
    ori_serlize_m_dict = dict([(score, (kw, ))] + list(ori_serlize_m_dict.items()))
    
    ori_serlize_m_dict = dict(list(sorted(ori_serlize_m_dict.items(), key = lambda t2: t2[0]))[:retain_size])
    print("list length : {}".format(len(ori_serlize_m_dict)))
    
    with open(serlize_path, "wb") as f:
        pkl.dump(ori_serlize_m_dict, f)
    print("time consum : {}".format(time() - start_time))
    '''
    
    return score


# In[25]:


best_fit_score = object_func(best_kw)


# In[26]:


best_fit_score


# In[ ]:




