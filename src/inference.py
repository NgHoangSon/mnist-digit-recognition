import torch

def predict(model, x):
    with torch.no_grad():
            logits=model(x)
            prob=torch.softmax(logits,dim=1)
            pred=prob.argmax(dim=1).item()
            confidence=prob[0,pred].item()
    return pred,confidence,prob