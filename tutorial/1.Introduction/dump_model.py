from transformers import AutoModelForMaskedLM
import torch


model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
print(model) # 印出模型結構

# fix old format of the model for candle start
model.eval()
output_stat_dict = model.state_dict() # 取得模型所有 weight
for key in output_stat_dict:
    if "beta" in key or "gamma" in key:
        print("warning: old format name:", key)
torch.save(output_stat_dict, "fix-bert-base-chinese.pth")
# fix old format of the model for candle end