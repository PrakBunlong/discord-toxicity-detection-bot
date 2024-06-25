import torch
from main import ToxicityClassifier, config, X_test, y_test

model = ToxicityClassifier(config)
model.load_state_dict(torch.load("./model.pth"))
print(model.eval())