import sys
sys.path.append('/shared/pso/step_5.1_analyze_outputs/funcs/plot_other.py')
import pandas as pd
import copy
import numpy as np
import warnings
from statsmodels.formula.api import ols
from plot_other import plot_other

class averages_and_error:
    def __init__(self):
        warnings.filterwarnings('ignore')
    def get_all_averages_and_error(self,timeseries_info,le_obs,strm_obs,
                                   start_err,end_err):
        to_load = list(timeseries_info.keys())
        for l,load in enumerate(to_load):
            # get the pixel-averaged and error information for pixel-scale outputs
            pixel_avgs = self.var_avgs(
                timeseries_info[load]['pixel_raw_timeseries']
            )
            timeseries_info[load]['pixel_avgs'] = pixel_avgs
            # get the watershed-averaged and error information for watershed-scale
            # ouptts
            wat_avgs = self.var_avgs(
                timeseries_info[load]['wat_raw_timeseries']
            )
            timeseries_info[load]['wat_avgs'] = wat_avgs
            this_le_err_df = self.var_error(
                timeseries_info[load]['pixel_raw_timeseries']['le'],
                le_obs,start_err,end_err
            )
            timeseries_info[load]['pixel_le_errors'] = this_le_err_df
            this_wat_err_df = self.var_error(
                timeseries_info[load]['wat_raw_timeseries']['strm_yr'],
                strm_obs,start_err,end_err
            )
            timeseries_info[load]['wat_strm_errors'] = this_wat_err_df
        return timeseries_info
    def var_avgs(self,raw_timeseries):
        '''
        Function that computes the average for each variable passed in the
        raw_timeseries dict
        Inputs:
            raw_timeseries: a dictionary of pd DataFrames, where each
            DataFrame is a different variable
        Outputs:
            avg_timeseries: a pandas df where each column is a pixel,
            each row is a different variables
        '''
        variables = list(raw_timeseries.keys())
        cols = list(raw_timeseries[variables[0]].columns)
        avgs_df = pd.DataFrame(columns=cols)
        for v,var in enumerate(variables):
            var_avgs = np.zeros(len(cols))
            this_df = raw_timeseries[var]
            for c,col in enumerate(cols):
                this_vals = this_df[col]
                this_avg = this_vals.mean()
                var_avgs[c] = this_avg
            avgs_df.loc[var] = var_avgs
        return avgs_df
    def var_error(self,preds,obs,start,end):
        '''
        Function that computes the error of the provided varaible. Computes a
        variety of different metrics and returns a DataFrame of the different
        metrics for each individual pixel and the average of these metrics.
        Inputs:
            preds: A DataFrame of the predictions
            obs: A DataFrame of the obs
            var_name: string that is the name of the variable of interest
            start: the date we want to start computing error
            end: the date we wnat to end computing error (inclusive)
        Ouputs:
            out_df: DataFrame where each column is a different pixel/basin and
            each row is a different error metric
        '''
        # get the dates that we want for both
        preds = preds.loc[start:end]
        obs = obs.loc[start:end]
        # let's only use columns where we have data at least 50% of the time
        col_nan_count = obs.isna().sum(axis=0)
        num_times = obs.shape[0]
        perc_nan = col_nan_count/num_times
        nan_cols = perc_nan.index[perc_nan > 0.5].to_list()
        obs[nan_cols] = np.nan
        # set the dataframe to store outputs
        cols = list(obs.columns)
        error_df = pd.DataFrame(columns=cols)
        # get the metrics
        rmse = self.rmse(preds,obs)
        error_df.loc['rmse'] = rmse
        ubrmse = self.ubrmse(preds,obs)
        error_df.loc['ubrmse'] = ubrmse
        r2 = self.r2(preds,obs)
        error_df.loc['r2'] = r2
        corr = self.corr(preds,obs)
        error_df.loc['corr'] = corr
        mae = self.mae(preds,obs)
        error_df.loc['mae'] = mae
        ubmae = self.ubmae(preds,obs)
        error_df.loc['ubmae'] = ubmae
        return error_df
    def rmse(self,preds,obs):
        diff = preds - obs
        rmse = diff**2
        rmse = rmse.mean(axis=0)
        rmse = rmse**0.5
        return rmse
    def ubrmse(self,preds,obs):
        unbiased_preds = preds - preds.mean(axis=0)
        unbiased_obs = obs - obs.mean(axis=0)
        diff = preds - obs
        ubrmse = diff**2
        ubrmse = ubrmse.mean(axis=0)
        ubrmse = ubrmse**0.5
        return ubrmse
    def r2(self,preds,obs):
        diff = preds - obs
        numerator = diff**2
        numerator = numerator.sum(axis=0)
        denominator = obs - obs.mean(axis=0)
        denominator = denominator**2
        denominator = denominator.sum(axis=0)
        r2 = 1 - numerator/denominator
        return r2
    def corr(self,preds,obs):
        avg_preds = preds.mean(axis=0)
        avg_obs = obs.mean(axis=0)
        numerator = (preds-avg_preds)*(obs-avg_obs)
        numerator = numerator.sum(axis=0)
        denominator_1 = (preds - avg_preds)**2
        denominator_1 = denominator_1.sum(axis=0)
        denominator_2 = (obs - avg_obs)**2
        denominator_2 = denominator_2.sum(axis=0)
        denominator = denominator_1*denominator_2
        denominator = denominator**0.5
        corr = numerator/denominator
        return corr
    def mae(self,preds,obs):
        diff = preds - obs
        mae = diff.abs()
        mae = mae.mean(axis=0)
        return mae
    def ubmae(self,preds,obs):
        unbiased_preds = preds - preds.mean(axis=0)
        unbiased_obs = obs - obs.mean(axis=0)
        diff = unbiased_preds - unbiased_obs
        ubmae = diff.abs()
        ubmae = ubmae.mean(axis=0)
        return ubmae
    def get_timestep_error(self,preds,obs,start,end):
        preds = preds.loc[start:end]
        obs = obs.loc[start:end]
        err_df = preds - obs
        abs_err_df = err_df.abs()
        cols = list(preds.columns)
        err_dict = {
            'err':err_df,
            'abs_err':abs_err_df
        }
        return err_dict
    def get_add_and_mult_bias(self,preds,obs,plots_dir,exp_name,
                              start='none',end='none',plot=False):
        # trim to start and end
        if start != 'none':
            preds = preds.loc[start:]
            obs = obs.loc[start:]
        if end != 'none':
            preds = preds.loc[start:]
            obs = obs.loc[:end]
        # check if everything is nan
        pred_nan_idx = np.where(np.isnan(preds) == True)[0]
        num_pred_nan = len(pred_nan_idx)
        obs_nan_idx = np.where(np.isnan(obs) == True)[0]
        num_obs_nan = len(obs_nan_idx)
        num_vals = len(preds)
        if num_pred_nan == num_vals or num_obs_nan == num_vals:
            add_bias = np.nan
            mult_bias = np.nan
            return [add_bias,mult_bias]
        # convert to flat np array
        preds_np = np.array(preds)
        preds_np_flat = preds_np.flatten()
        obs_np = np.array(obs)
        obs_np_flat = obs_np.flatten()
        # put in df
        data_df = pd.DataFrame(
            {
                'preds':preds_np_flat,
                'obs':obs_np_flat
            }
        )
        # create teh model
        model = ols(
            "preds ~ obs",
            data_df
        ).fit()
        coefs = model._results.params
        add_bias = coefs[0]
        mult_bias = coefs[1]
        if plot:
            scatter_save_name = (
                'error_model_scatter_{}.png'.format(exp_name)
            )
            plot = plot_other()
            plot.scatter(
                obs_np_flat,preds_np_flat,plots_dir,scatter_save_name,
                'GLEAM LE obs (W/m2)',
                '{} LE preds (W/m2)'.format(exp_name),
                best_fit_line=True,
                dot_size=0.5
            )
        return [add_bias,mult_bias]
















