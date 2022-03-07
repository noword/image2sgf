import torch
weights = torch.load('weiqi.pth', map_location=torch.device('cpu'))
weights['roi_heads.box_predictor.cls_score.weight'] = weights['roi_heads.box_predictor.cls_score.weight'][:5]
weights['roi_heads.box_predictor.cls_score.bias'] = weights['roi_heads.box_predictor.cls_score.bias'][:5]
weights['roi_heads.box_predictor.bbox_pred.weight'] = weights['roi_heads.box_predictor.bbox_pred.weight'][:20]
weights['roi_heads.box_predictor.bbox_pred.bias'] = weights['roi_heads.box_predictor.bbox_pred.bias'][:20]

torch.save(weights, 'weiqi_board.pth')
