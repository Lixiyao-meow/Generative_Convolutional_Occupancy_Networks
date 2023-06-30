import torch

path = "./out/IntrA_encoder_decoder_6.26/model_best.pt"
cp = torch.load(path)
print(cp.items())