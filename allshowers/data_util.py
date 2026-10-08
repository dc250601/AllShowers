### After cleaning the mess...
### Author: Diptarko Choudhury
import torch
from .data_sets import batched_histogram, ModelInputDict
from .preprocessing import *
import showerdata
import numpy as np

class ShowerDatasetFromH5(torch.utils.data.Dataset):

    def __init__(self,
                 path,
                 energy_scaler,
                 coordinate_scaler,
                 conditioning_scaler,
                 num_layers,
                 RESIZE_FACTOR = 1,
                 start = None,
                 stop = None
                ):

        self.path = path
        self.num_layers = num_layers
        
        self.GlobalStart = start if start is not None else 0
        self.GlobalStop = stop if stop is not None else showerdata.get_file_length(self.path)
    
        self.coordinate_scaler = coordinate_scaler
        self.energy_scaler = energy_scaler
        self.conditioning_scaler = conditioning_scaler
        self.RESIZE_FACTOR = RESIZE_FACTOR
        
        self.__prepare_label_look_up_table()
        
    def __len__(self):
        return self.GlobalStop - self.GlobalStart

    def __prepare_label_look_up_table(self):
        global_pdg = showerdata.observables.read_observables_from_file(self.path,
                                                               ["incident_pdg"],
                                                               start = self.GlobalStart,
                                                               stop = self.GlobalStop)
        global_pdg = global_pdg["incident_pdg"]
        unique_pdg = np.unique(global_pdg).tolist()
        unique_pdg.sort(key=lambda x: (abs(x), -x))
        label_list = unique_pdg
        self.look_up = {label_list[i]: i for i in range(len(unique_pdg))}

    def __getitem__(self,
                     idx):
        
        start = self.GlobalStart + idx
        stop = self.GlobalStart + idx+1
        
        shower = showerdata.load(self.path,
                             start = start,
                             stop = stop)
        
        energy = torch.from_numpy(shower.energies)
        direction = torch.from_numpy(shower.directions)
        pdg = shower.pdg
        shower =torch.from_numpy(shower.points)
        
        noise,_ = showerdata.load_target(self.path,
                                   "target",
                                   start=start,
                                   stop=stop)
         
        mask = shower[:,:,[3]] > 0
        energy = self.conditioning_scaler(self.RESIZE_FACTOR*energy)
        
        coordinates_xy = self.coordinate_scaler(shower[:,:,[0,1]])
        hit_energy = self.energy_scaler(shower[:,:,[3]])
        
        x = torch.concat([coordinates_xy,hit_energy],dim=-1)
        x[~mask.repeat(1,1,3)] = 0.0
        
        layer = (shower[:,:,[2]] + 0.1).long()   ### Why was this 0.1 added ?
        
        num_points = batched_histogram(
                    data=layer.squeeze(dim=-1),
                    mask=mask.squeeze(dim=-1),
                    num_bins=self.num_layers,
                )
        
        label = self.look_up[pdg.item()]
    
        cond = torch.concatenate([energy,direction],dim=-1)
        
        return ModelInputDict(
            x = x.squeeze(0),
            cond = cond.squeeze(0),
            num_points = num_points.squeeze(0),
            layer = layer.squeeze(0),
            mask = mask.squeeze(0),
            label = label,
            noise = noise.squeeze(0)
        )

    @staticmethod
    def PrepareScalesFromTrafoFile(config_dataset,
                                   trafo_path,
                                   rank,
                                   world_size,
                                   local_rank
                                  ):

        if world_size > 1:
            torch.distributed.barrier(device_ids=[local_rank])
            
        config_dataset = config_dataset.copy()
        all_scalers = torch.load(trafo_path)

        energy_scaler = compose(config_dataset["samples_energy_trafo"])
        coordinate_scaler = compose(config_dataset["samples_coordinate_trafo"])
        conditioning_scaler = compose(config_dataset["cond_trafo"])

        energy_scaler.load_state_dict(all_scalers["samples_energy_trafo"])
        coordinate_scaler.load_state_dict(all_scalers["samples_coordinate_trafo"])
        conditioning_scaler.load_state_dict(all_scalers["cond_trafo"])

        if world_size > 1:
            torch.distributed.barrier(device_ids=[local_rank])

        return energy_scaler, coordinate_scaler, conditioning_scaler
                