import os
import datetime
import numpy as np
import copy
from tqdm import tqdm

from Model.CNN_Transformer import TransformerNN as TransformerNN

from TensorProcessing.Dataset import FourWellDataset, OfflineGeoDataset
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


class MyProblem(Problem):
    def __init__(self, robust_optimization):
        super().__init__(n_var=160,
                         n_obj=2,
                         n_ieq_constr=1,
                         xl=4.,
                         xu=8.)

        self.robust_optimization = robust_optimization
        self.device = 'cuda'

    def _evaluate(self, x, out, *args, **kwargs):
        # x shape: [N_pop, n_var] --> [N_pop, 40, 4] for 4 wells
        # then send to Surrogate to generate P/S of size [N_pop, t, z, x, y]
        # then calculate the two objective functions

        x = x.astype(np.float32)
        f1, f_thres_max = self.pressure_evaluate(x)  # [N_pop, ]
        f2 = self.saturation_evaluate(x)  # [N_pop, ]

        out["F"] = np.column_stack([f1, f2])  # [N_pop, 2]

        v1, v2 = self.total_inj_constraint(x)
        g2 = 24 - v2  # mean rate of all 4 wells averaged over 40 time steps >= 24 kg /s
        out["G"] = g2

    def total_inj_constraint(self, flatten_rate):
        reshaped_rate = copy.deepcopy(flatten_rate).reshape(-1, 40, 4)  # [N_pop, 40, 4]
        four_well_sum = np.sum(reshaped_rate, axis=-1)  # [N_pop, 40]
        sum_min = np.min(four_well_sum, axis=-1)  # the min value of 4 well rate sum
        sum_mean = np.mean(four_well_sum, axis=-1)  # the mean value of 4 well rate sum
        return sum_min, sum_mean

    def rate_constraint(self, flatten_rate):
        reshaped_rate = copy.deepcopy(flatten_rate).reshape(-1, 40, 4)  # [N_pop, 40, 4]
        max_rate_diff = []
        for well_idx in range(reshaped_rate.shape[-1]):
            well_rate = reshaped_rate[:, :, well_idx]  # [N_pop, 40]
            well_rate_diff = np.abs(np.diff(well_rate, axis=1))  # [N_pop, ]
            max_rate_diff.append(np.max(well_rate_diff, axis=1))
        max_rate_diff = np.array(max_rate_diff)
        return np.max(max_rate_diff, axis=0)

    def pressure_evaluate(self, flatten_rate):
        input_geo = self.wrap_dataset().to(self.device)  # [1 or 50, z, x, y]
        reshaped_rate = copy.deepcopy(flatten_rate).reshape(-1, 40, 4)  # [N_pop, 40, 4]
        reshaped_rate = (reshaped_rate - 4) / 4
        input_rate = torch.tensor(reshaped_rate).to(self.device)
        surrogate_model = self.pressure_surrogate()

        num_realizations = input_geo.size(0)
        num_populations = input_rate.size(0)
        thres_case_list = []

        for i_case in range(num_realizations):
            tmp_input_geo = input_geo[i_case].unsqueeze(0).repeat(num_populations, 1, 1, 1)
            tmp_input_rate = input_rate

            with torch.no_grad():
                P_pred, _ = surrogate_model(tmp_input_geo, tmp_input_rate)  # [b, t, z, x, y]

            P_pred = (P_pred * (40690440.0 - 18954842.0) + 18954842.0)
            b, t, z, x, y = P_pred.shape
            init_pressure = torch.tensor(
                [19417956.0, 19358242.0, 19298532.0, 19238822.0, 19179114.0, 19119408.0, 19059704.0, 19000000.0],
                dtype=torch.float32).reshape(z, 1, 1).to(self.device)
            init_pressure = init_pressure.expand(z, x, y)
            init_pressure = init_pressure.unsqueeze(0).unsqueeze(0).expand(b, t, z, x, y)

            over_pressure = (P_pred - init_pressure) / 1e6

            P_flatten = torch.flatten(over_pressure, start_dim=1)
            P_max, _ = torch.max(P_flatten, dim=1)
            P_max = P_max.detach().cpu().tolist()
            thres_case_list.append(P_max)

        thres_case_list = np.array(thres_case_list)  # [num_realizations, N_pop]
        P_thres_mean, P_thres_max = np.mean(thres_case_list, axis=0), np.max(thres_case_list, axis=0)
        return P_thres_mean, P_thres_max

    def saturation_evaluate(self, flatten_rate):
        input_geo = self.wrap_dataset().to(self.device)  # [1 or 50, z, x, y]
        reshaped_rate = copy.deepcopy(flatten_rate).reshape(-1, 40, 4)  # [N_pop, 40, 4]
        reshaped_rate = (reshaped_rate - 4) / 4
        input_rate = torch.tensor(reshaped_rate).to(self.device)
        surrogate_model = self.saturation_surrogate()

        num_realizations = input_geo.size(0)
        num_populations = input_rate.size(0)
        obj_case_list = []

        for i_case in range(num_realizations):
            tmp_input_geo = input_geo[i_case].unsqueeze(0).repeat(num_populations, 1, 1, 1)
            tmp_input_rate = input_rate
            with torch.no_grad():
                S_pred, _ = surrogate_model(tmp_input_geo, tmp_input_rate)
            obj_case = self.cal_sat_obj(S_pred)
            obj_case_list.append(obj_case)

        obj_case_list = np.array(obj_case_list)  # [num_realizations, N_pop]
        E_obj = np.mean(obj_case_list, axis=0)  # [N_pop, ]
        return -E_obj

    def cal_pre_obj(self, pred_pressure_tensor):
        uip = torch.std(pred_pressure_tensor, dim=(2, 3, 4))  # [N_pop, t]
        mean_uip = torch.mean(uip, dim=1)  # [N_pop, ]
        return mean_uip.detach().cpu().tolist()

    def cal_sat_obj(self, pred_saturation_tensor):
        pred_saturation_tensor = torch.where(pred_saturation_tensor <= 0.01,
                                             torch.tensor(0.0, dtype=pred_saturation_tensor.dtype,
                                                          device=pred_saturation_tensor.device), pred_saturation_tensor)
        storage_percentage = torch.sum(pred_saturation_tensor, dim=(2, 3, 4))  # [N_pop, t]
        metric_saturation = storage_percentage[:, -1]  # [N_pop, ]
        metric_saturation = metric_saturation / (8 * 40 * 40 * 0.89) * 100
        return metric_saturation.detach().cpu().tolist()

    def pressure_surrogate(self):
        val_model_pre = TransformerNN(
            transformer_embed_size=200,
            transformer_target_size=200,
            transformer_num_layers=2,
            transformer_heads=8,
            transformer_forward_expansion=4,
            transformer_dropout=0.0,
            transformer_dt=86400 * 30 * 6,
            transformer_num_ts=40,
            device='cuda',
            geo_in_channel=8,
            geo_embed_size=200,
            rate_embed_size=200,
            decoder_channel=40
        ).to(self.device)
        val_model_pre.load_state_dict(
            torch.load(f'G:\\optim_code\\checkpoints\\ablations\\base\\pressure_self_add_2layer_8head.pth',
                       map_location=self.device))
        val_model_pre.eval()
        return val_model_pre

    def saturation_surrogate(self):
        val_model_sat = TransformerNN(
            transformer_embed_size=200,
            transformer_target_size=200,
            transformer_num_layers=2,
            transformer_heads=8,
            transformer_forward_expansion=4,
            transformer_dropout=0.0,
            transformer_dt=86400 * 30 * 6,
            transformer_num_ts=40,
            device='cuda',
            geo_in_channel=8,
            geo_embed_size=200,
            rate_embed_size=200,
            decoder_channel=40
        ).to(self.device)
        val_model_sat.load_state_dict(
            torch.load(f'G:\\optim_code\\checkpoints\\ablations\\base\\saturation_self_add_2layer_8head.pth',
                       map_location=self.device))
        val_model_sat.eval()
        return val_model_sat

    def wrap_dataset(self):
        val_dataset_sat = FourWellDataset(train_or_val='test', label_name='saturation')
        val_loader_sat = DataLoader(val_dataset_sat, batch_size=len(val_dataset_sat), shuffle=False, drop_last=False,
                                    num_workers=0, pin_memory=False)

        data_iter_sat = iter(val_loader_sat)
        x_geo_sat, _, _ = next(data_iter_sat)
        if self.robust_optimization:
            x_geo = x_geo_sat[250:350]  # an ensemble of 50 realizations
        else:
            x_geo = x_geo_sat[295:296]  # only 1 deterministic realization
        # print(f'Loaded Geo Parameter size: {x_geo.shape}')
        return x_geo


