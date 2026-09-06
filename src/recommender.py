import torch
import torch.nn as nn
import torch.nn.functional as F

class DriverTower(nn.Module):
    def __init__(self, num_zones=263, embed_dim=16):
        super(DriverTower, self).__init__()
        self.zone_embed = nn.Embedding(num_zones + 1, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, current_zone_id, hour, shift_duration):
        z_emb = self.zone_embed(current_zone_id)
        context = torch.cat([z_emb, hour.unsqueeze(1), shift_duration.unsqueeze(1)], dim=1)
        return self.fc(context)


class ZoneTower(nn.Module):
    def __init__(self, num_zones=263, embed_dim=16):
        super(ZoneTower, self).__init__()
        self.zone_embed = nn.Embedding(num_zones + 1, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, candidate_zone_id, predicted_demand, avg_fare):
        z_emb = self.zone_embed(candidate_zone_id)
        context = torch.cat([z_emb, predicted_demand.unsqueeze(1), avg_fare.unsqueeze(1)], dim=1)
        return self.fc(context)


class TwoTowerTaxiRecommender(nn.Module):
    def __init__(self, num_zones=263):
        super(TwoTowerTaxiRecommender, self).__init__()
        self.driver_tower = DriverTower(num_zones=num_zones)
        self.zone_tower = ZoneTower(num_zones=num_zones)

    def forward(self, driver_inputs, zone_inputs):
        driver_emb = self.driver_tower(*driver_inputs)
        zone_emb = self.zone_tower(*zone_inputs)
        
        driver_emb = F.normalize(driver_emb, p=2, dim=1)
        zone_emb = F.normalize(zone_emb, p=2, dim=1)
        
        scores = torch.sum(driver_emb * zone_emb, dim=1)
        return scores