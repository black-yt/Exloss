import torch
from torch.functional import F

def Exloss(pred, target, up_th=0.9, down_th=0.1, lamda_underestimate=2.5, lamda_overestimate=0.5, lamda=1.0):
    '''
    up_th: 极大值的百分位阈值
    down_th: 极小值的百分位阈值
    lamda_underestimate: 低估时的惩罚，大于高估时的惩罚
    lamda_overestimate: 高估时的惩罚
    lamda: Exloss与MSE的权重
    '''
    
    mse_loss = torch.mean((pred-target)**2)

    N, C, H, W = pred.shape
    # 得到 target 中 90% 和 10% 的分位点作为极端最大值和极端最小值的阈值，记作 tar_up 和 tar_down
    tar_up =  torch.quantile(target.view(N, C, H*W), q=up_th, dim=-1).unsqueeze(-1).unsqueeze(-1) # N,C,1,1
    tar_down =  torch.quantile(target.view(N, C, H*W), q=down_th, dim=-1).unsqueeze(-1).unsqueeze(-1) # N,C,1,1

    target_up_area = F.relu(target-tar_up) # target 中大于 tar_up 的部分
    target_down_area = -F.relu(tar_down-target) # target 中小于 tar_down 的部分
    pred_up_area = F.relu(pred-tar_up) # pred 中大于 tar_up 的部分
    pred_down_area = -F.relu(tar_down-pred) # pred 中小于 tar_down 的部分

    # 对于 pred 中低估（极大值预测偏小，极小值预测偏大）的部分加大 loss 权重
    loss_up = lamda_underestimate*(target_up_area-pred_up_area)*F.relu(target_up_area-pred_up_area)+\
              lamda_overestimate*(pred_up_area-target_up_area)*F.relu(pred_up_area-target_up_area)
    loss_down = lamda_overestimate*(target_down_area-pred_down_area)*F.relu(target_down_area-pred_down_area)+\
                lamda_underestimate*(pred_down_area-target_down_area)*F.relu(pred_down_area-target_down_area)
    loss_up = torch.mean(loss_up)
    loss_down = torch.mean(loss_down)
    ex_loss = (loss_up + loss_down)/(1-up_th+down_th)

    loss_all = mse_loss + lamda*ex_loss

    # print("all_loss:", loss_all.item(), "mse_loss:", mse_loss.item(), "ex_loss", ex_loss.item())

    return loss_all

if __name__ == "__main__":
    pred = torch.randn(1,69,721,1440)
    target = torch.randn(1,69,721,1440)
    print(Exloss(pred, target))