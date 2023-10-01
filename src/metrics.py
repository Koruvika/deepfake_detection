def compute_accuray(prediction, target):
    pred_idx = prediction.argmax(dim=1).cpu().data.numpy()
    tmp = pred_idx == target.cpu().numpy()
    return sum(tmp) / len(pred_idx)
