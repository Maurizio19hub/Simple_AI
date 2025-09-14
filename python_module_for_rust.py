import os
import json
import numpy as np

NN = [784, 128, 64, 10]

def softmax(z):
    z = z - np.max(z)            # stabilità numerica
    e = np.exp(z)
    return e / np.sum(e)

def cross_entropy_loss(a, y):
    return -np.sum(y * np.log(a + 1e-12))


def sigmoidFun(x):
    return 1 / (1 + np.exp(-x))

def testingFun(x : int) -> str:
    print("Sono python")
    net = "../net_trained.json"
    if os.path.exists(net) and os.path.getsize(net) > 0:
        with open(net) as f:
            Net = json.load(f)
        f.close()
        for layer in Net:
            Net[layer]["w"] = np.array(Net[layer]["w"])
            Net[layer]["b"] = np.array(Net[layer]["b"])
            Net[layer]["a"] = np.array(Net[layer]["a"])

    print("Rete caricata.")
    image = np.array(x)

    
    
    Net[f'layer{0}'] = {}    
    Net[f'layer{0}']["a"] = image
    for r in range(1, len(NN)):
        for i in range(NN[r]):
            sum = 0
            for weight, bit in zip(Net[f'layer{r}']["w"][:, i], Net[f'layer{r-1}']["a"]):
                sum += weight * bit
            sum += Net[f'layer{r}']["b"][i]
            if r == len(NN) - 1:
                Net[f'layer{r}']["a"][i] = sum #softmax per ultimo layer
            else:
                Net[f'layer{r}']["a"][i] = sigmoidFun(sum)
    Net[f'layer{len(NN)-1}']["a"] = softmax(Net[f'layer{len(NN)-1}']["a"]) #softmax per ultimo layer  
    
    index = np.argmax(Net[f'layer{len(NN)-1}']['a'])
    
    print("final layer : ", Net[f'layer{len(NN)-1}']['a'])
    return f'Il numero disegnato è : {index}'