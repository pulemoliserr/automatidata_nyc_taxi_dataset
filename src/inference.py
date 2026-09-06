# src/inference.py
import torch

def recommend_top_k_zones(driver_state, candidate_zones, stgnn_model, recommender_model, adj_matrix, k=5):
    stgnn_model.eval()
    recommender_model.eval()
    
    with torch.no_grad():
        # 1. Forecast demand with GNN
        raw_forecasts = stgnn_model(driver_state['historical_seq'], adj_matrix)
        
        # Force flatten to a pure 1D vector of length 263
        demand_forecasts = raw_forecasts.view(-1)
        
        # 2. Extract Candidate Zone features
        candidate_ids = torch.tensor([z['id'] for z in candidate_zones], dtype=torch.long)
        avg_fares = torch.tensor([z['avg_fare'] for z in candidate_zones], dtype=torch.float32)
        
        # Safely extract matching zone demands from the 1D vector
        zone_demands = demand_forecasts[candidate_ids - 1]
        
        # 3. Broadcast Driver Inputs across all candidate zones
        num_candidates = len(candidate_zones)
        d_zone = torch.tensor([driver_state['zone_id']] * num_candidates, dtype=torch.long)
        d_hour = torch.tensor([driver_state['hour']] * num_candidates, dtype=torch.float32)
        d_shift = torch.tensor([driver_state['shift_duration']] * num_candidates, dtype=torch.float32)
        
        # 4. Two-Tower Forward Pass
        scores = recommender_model(
            driver_inputs=(d_zone, d_hour, d_shift),
            zone_inputs=(candidate_ids, zone_demands, avg_fares)
        )
        
        # Ensure scores are also a 1D tensor
        scores = scores.view(-1)
        
        # 5. Extract Top K Candidates
        top_k_indices = torch.topk(scores, k=k).indices.numpy()
        
        recommendations = [
            {
                "zone_id": candidate_zones[idx]['id'],
                "score": round(float(scores[idx]), 4),
                "predicted_demand": round(float(zone_demands[idx]), 1)
            }
            for idx in top_k_indices
        ]
        
    return recommendations