import torch

def get_top_k_recommendations(model, user_id, n_items, k, device, seen_items=None):
    
    user_tensor = torch.tensor([user_id] * n_items).to(device)
    item_tensor = torch.arange(n_items).to(device)

    with torch.no_grad():
        scores = model(user_tensor, item_tensor)
    
    scores = scores.cpu()

    if seen_items is not None:
        scores[seen_items] = -1e9

    top_k = torch.topk(scores, k=k).indices.numpy()

    return top_k