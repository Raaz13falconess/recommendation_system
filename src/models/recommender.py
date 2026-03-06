import torch

def get_top_k_recommendations(model, user_id, n_items, k, device):
    
    user_tensor = torch.tensor([user_id] * n_items).to(device)
    item_tensor = torch.arange(n_items).to(device)

    with torch.no_grad():
        scores = model(user_tensor, item_tensor)
    
    top_k = torch.topk(scores, k=k).indices.cpu().numpy()

    return top_k