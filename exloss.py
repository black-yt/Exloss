import torch
from torch.functional import F

def Exloss(pred, target):
    mse_loss = torch.mean((pred-target)**2)

    N, C, H, W = pred.shape
    # 得到target中 90% 和 10% 的分位点作为极端最大值和极端最小值的阈值
    up_th = 0.9
    down_th = 0.1
    tar_up =  torch.quantile(target.view(N, C, H*W), q=up_th, dim=-1).unsqueeze(-1).unsqueeze(-1) # N,C,1,1
    tar_down =  torch.quantile(target.view(N, C, H*W), q=down_th, dim=-1).unsqueeze(-1).unsqueeze(-1) # N,C,1,1

    target_up_area = F.relu(target-tar_up) # target中大于极端最大值阈值的部分
    target_down_area = -F.relu(tar_down-target) # target中小于极端最小值阈值的部分
    pred_up_area = F.relu(pred-tar_up) # pred中大于极端最大值阈值的部分
    pred_down_area = -F.relu(tar_down-pred) # pred中小于极端最小值阈值的部分

    # 对于pred中低估的部分加大loss
    lamda_underestimate = 2.5 # 低估时的loss权重，加大惩罚
    lamda_overestimate = 0.5 # 高估时的loss权重
    loss_up = lamda_underestimate*(target_up_area-pred_up_area)*F.relu(target_up_area-pred_up_area)+\
              lamda_overestimate*(pred_up_area-target_up_area)*F.relu(pred_up_area-target_up_area)
    loss_down = lamda_overestimate*(target_down_area-pred_down_area)*F.relu(target_down_area-pred_down_area)+\
                lamda_underestimate*(pred_down_area-target_down_area)*F.relu(pred_down_area-target_down_area)
    loss_up = torch.mean(loss_up)
    loss_down = torch.mean(loss_down)
    ex_loss = (loss_up + loss_down)/(1-up_th+down_th)

    loss_all = mse_loss + ex_loss

    print("all_loss:", loss_all.item(), "mse_loss:", mse_loss.item(), "ex_loss", ex_loss.item())

    return loss_all

if __name__ == "__main__":
    pred = torch.randn(1,69,721,1440)
    target = torch.randn(1,69,721,1440)
    print(Exloss(pred, target))