import numpy as np
import json
import struct
import os

NN = [784, 128, 64, 10]

def createWeights() -> dict:
    Net = dict()
    for i in range(1, len(NN)):
        std = np.sqrt(2/(NN[i-1]+NN[i]))
        w = np.zeros((NN[i-1], NN[i]))
        b = np.zeros(NN[i])
        a = np.zeros(NN[i])
        for x in range(NN[i-1]):
            for y in range(NN[i]):
                w[x][y] = np.random.randn()*std
        Net[f'layer{i}'] = {}
        Net[f'layer{i}']["w"] = w
        Net[f'layer{i}']["b"] = b
        Net[f'layer{i}']["a"] = a
        
    Net_json = {}
            
    for key, value in Net.items():
        Net_json[key] = {
            "w" : value["w"].tolist(),
            "b" : value["b"].tolist(),
            "a" : value["a"].tolist()
        }

    with open("net.json", "w") as f:
        json.dump(Net_json, f)
    
    return Net


def gradientFun() -> dict:
    Grad = dict()
    for i in range(1, len(NN)):
        w = np.zeros((NN[i-1], NN[i]))
        b = np.zeros(NN[i])
        Grad[f'layer{i}'] = {}
        Grad[f'layer{i}']["w"] = w
        Grad[f'layer{i}']["b"] = b
    return Grad

def loadImages():
    with open("train-images-idx3-ubyte/train-images.idx3-ubyte", "rb") as f:
        #headers
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        #images
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows * cols)
        #needs to be from 0 to 1
        images = images.astype(np.float32) / 255.0
        return images

def loadLabels():
    with open("train-labels-idx1-ubyte/train-labels.idx1-ubyte", "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        one_hot = np.zeros((num_labels, 10))
        one_hot[np.arange(num_labels), labels] = 1
        return one_hot
    
def batchGenerator(batch_size, images, labels):
    #for loop on num_imgages
    for i in range(0, images.shape[0], batch_size):
        yield images[i : i + batch_size], labels[i : i + batch_size]
    
def backPropDelta(a, one_hot, h, prev_delta, Net):
    if h == len(NN) - 1:
        delta = 2 * (a - one_hot) * a * (np.ones(NN[h]) - a) #assicurarsi che sia una operazione tra vettori prima di moltiplicare per a(N-1)
    else:
        delta = np.dot(Net[f'layer{h+1}']["w"], prev_delta) * (1 - a) * a #per layer intermedi
        #print(delta.shape)
    return delta


def sigmoidFun(x):
    return 1 / (1 + np.exp(-x))

def lossFun(x, y):
    '''
    sum = 0
    for bit_x, bit_y in zip(x, y):
        sum += pow((bit_x - bit_y), 2)
    return sum'''
    return np.sum(np.power((x - y), 2))

def testingFun():
    pass

def main():
    scelta = int(input("Selezionare cosa si vuole fare:"
    "   1) Fase di addestramento"
    "   2) Fase di testing"))
    if scelta == 2:
        testingFun()
        return
    elif scelta != 1:
        print("Scelta non valida.")
        return

    #se i pesi sono gia salvati nel file json, allora procedo a prelevarli, altrimenti li genero
    if os.path.exists("net.json") and os.path.getsize("net.json") > 0:
        with open("net.json", "r") as f:
            Net = json.load(f)
        f.close()
        for layer in Net:
            Net[layer]["w"] = np.array(Net[layer]["w"])
            Net[layer]["b"] = np.array(Net[layer]["b"])
            Net[layer]["a"] = np.array(Net[layer]["a"])
        print("Rete caricata dal file net.json.")
    else:
        Net = createWeights()
        print("Rete creata.")

    images = loadImages()
    labels = loadLabels()
    batch_size = 32
    #print(labels)

    #Creazione del Gradiente
    batchGrad = gradientFun()

    
    
    #BATCH IMMAGINI
    batch_counter = 0
    for images_batch, labels_batch in batchGenerator(batch_size, images, labels):
        avg_loss = 0
        #SINGOLA IMMAGINE DEL BATCH
        for image, label in zip(images_batch, labels_batch):
            #image = images_batch[0]
            #label = labels_batch[0]

            Net[f'layer{0}'] = {}
            Net[f'layer{0}']["a"] = image
        
            #PER LA SINGOLA IMMAGINE CONSIDERO OGNI LAYER
            for r in range(1, len(NN)):
                #al passaggio al layer successivo devo cambiare image nel nuovo stato che sarà un vettore più corto e formato da sigmoidi
                #PER OGNI LAYER CONSIDERO OGNI NODO
                for i in range(NN[r]):
                    sum = 0
                    
                    for weight, bit in zip(Net[f'layer{r}']["w"][:, i], Net[f'layer{r-1}']["a"]):
                        '''if bit != 0.0:
                            print(weight, " ******* ", bit)
                        print(im_counter)'''
                        sum += weight * bit

                    sum += Net[f'layer{r}']["b"][i]
                    Net[f'layer{r}']["a"][i] = sigmoidFun(sum)

                    #print(f'layer{r} : {i} -> ', Net[f'layer{r}']["a"][i] )

            #ora che ho l'output della rete (layer3) posso calcolare le funzioni di costo per ogni immagine e poi fare la media di queste per il signolo batch
            #print(Net[f'layer{(len(NN)-1)}']["a"], label)
            delta = None
            for h in range(len(NN)-1, 0, -1):
                #print(h)
                prev_delta = delta
                var_a = Net[f'layer{h}']["a"]
                var_a_prev = Net[f'layer{h-1}']["a"]
                delta = backPropDelta(var_a, label, h, prev_delta, Net)
                batchGrad[f'layer{h}']["w"] += np.outer(var_a_prev, delta) #gradiente del layer corrente per w
                batchGrad[f'layer{h}']["b"] += delta
                #print((Net[f'layer{h}']["w"]).shape, delta.shape, (Net[f'layer{h-1}']["a"]).shape)
                

            loss_fun = lossFun(Net[f'layer{(len(NN)-1)}']["a"], label)
            avg_loss += loss_fun
            #print(loss_fun, avg_loss)
            #print(np.sum(np.power((Net[f'layer{(len(NN)-1)}']["a"] - label), 2)))
        
        avg_loss /= batch_size
        for layer in batchGrad:
            #print("***************", batchGrad[layer]["b"])
            batchGrad[layer]["w"] /= batch_size
            batchGrad[layer]["b"] /= batch_size
            #print(batchGrad[layer]["b"])
        #sembra workare, abbiamo ottenuto la media dei gradienti di w e b rispetto al batch

        #*******
        #adesso devo aggiustare i pesi con l'utilizzo dei gradienti medi e poi basta ripetere per tutti i batch
        '''print("########################")
        print(batchGrad[f'layer{1}']["w"].shape)
        print(batchGrad[f'layer{1}']["b"].shape)'''

        learning_rate = 0.01
        for layer in batchGrad:
            #print("£££££££££££££££££", Net[layer]["b"])     
            Net[layer]["w"] -= learning_rate * batchGrad[layer]["w"]
            Net[layer]["b"] -= learning_rate * batchGrad[layer]["b"]
            #print(Net[layer]["b"])
        #*******

        print(f'Batch calcolato : {batch_counter} ; con loss : {avg_loss}')
        batch_counter += 1
    
    Net_json = {}

    for r in range(1, len(NN)):
        Net_json[f'layer{r}'] = {
            "w" : Net[f'layer{r}']["w"].tolist(),
            "b" : Net[f'layer{r}']["b"].tolist(),
            "a" : Net[f'layer{r}']["a"].tolist()
        }

    with open("net_trained.json", "w") as f:
        json.dump(Net_json, f)
    print("File net_trained.json creato con i pesi e i bias aggiornati")
    return
    






if __name__ == "__main__":
    main()