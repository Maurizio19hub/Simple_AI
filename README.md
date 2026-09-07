# Simple_AI 🧠

Una rete neurale **completamente implementata in NumPy puro** (senza framework di Machine Learning come TensorFlow o PyTorch) per la classificazione del dataset **MNIST** di cifre scritte a mano.

Il progetto nasce come esercizio didattico per capire *da dentro* come funziona una rete neurale: forward pass, backpropagation, discesa del gradiente, mini-batch, funzioni di attivazione e di costo.

---

## 📐 Architettura della rete

La rete è un **Multi-Layer Perceptron (MLP)** fully-connected con la seguente struttura:

```
Input (784) ──► Hidden 1 (128) ──► Hidden 2 (64) ──► Output (10)
```

```python
NN = [784, 128, 64, 10]
```

| Layer | Neuroni | Attivazione |
|---|---|---|
| Input | 784 | — (pixel dell'immagine 28×28 appiattiti, normalizzati in [0,1]) |
| Hidden 1 | 128 | Sigmoide: `σ(x) = 1 / (1 + e^-x)` |
| Hidden 2 | 64 | Sigmoide |
| Output | 10 | Softmax (distribuzione di probabilità sulle 10 cifre) |

**Parametri totali:** (784×128 + 128) + (128×64 + 64) + (64×10 + 10) = **109.386** pesi e bias.

### Inizializzazione dei pesi

I pesi sono inizializzati con la tecnica di **Glorot/Xavier**:

```python
std = sqrt(2 / (n_input + n_output))
w = randn() * std
```

Questa scelta mantiene la varianza delle attivazioni stabile tra i layer, evitando l'esplosione o lo svanimento dei segnali all'inizio dell'addestramento.

### Funzione di costo

**Cross-Entropy Loss**, matematicamente coerente con la coppia Softmax + output one-hot:

```
L = -Σ y_i · log(a_i)
```

Grazie a questa combinazione, il delta dell'ultimo layer si semplifica in:

```python
delta = (a - one_hot)   # senza termini di derivata di attivazione
```

che è il motivo per cui nel codice `backPropDelta` ha un ramo speciale per l'ultimo layer (nel codice sono presenti anche i commenti con la variante per Sigmoide + Mean Squared Error).

### Backpropagation e ottimizzazione

- **Algoritmo:** Mini-Batch Gradient Descent con `batch_size = 32`.
- **Learning rate:** `0.1`.
- **Epoche:** `10` (`AGES = 10`).
- **Shuffling:** a inizio di ogni epoca il dataset viene mescolato per evitare correlazioni nell'ordine delle immagini.

Il gradiente per ogni batch viene **accumulato** sample per sample (`np.outer(a_prev, delta)` per i pesi, `delta` per i bias) e poi **mediato** prima dell'aggiornamento:

```
w ← w − η · (1/B) · Σ_batch ∇w
```

La propagazione dell'errore all'indietro per i layer nascosti usa la derivata della sigmoide:

```
delta_h = (W_{h+1} · delta_{h+1}) ⊙ a_h ⊙ (1 − a_h)
```

## 📁 Struttura del progetto

| File | Descrizione |
|---|---|
| `main.py` | Training e testing della rete: caricamento MNIST, forward pass, backprop, salvataggio pesi |
| `net.json` | Pesi inizializzati casualmente (creato alla prima generazione della rete) |
| `net_trained.json` | Pesi dopo l'addestramento, salvati a fine di ogni epoca |
| `train-images-idx3-ubyte/`, `train-labels-idx1-ubyte/` | MNIST training set (60.000 immagini) |
| `t10k-images-idx3-ubyte/`, `t10k-labels-idx1-ubyte/` | MNIST test set (10.000 immagini) |

## 🚀 Utilizzo

### Requisiti

- Python 3.x
- NumPy

```bash
pip install numpy
```

### Download del dataset

Scarica i 4 file MNIST in formato IDX (es. da [MNIST database](http://yann.lecun.com/exdb/mnist/), senza decomprimerli) e organizzali così:

```
Simple_AI/
├── train-images-idx3-ubyte/train-images.idx3-ubyte
├── train-labels-idx1-ubyte/train-labels.idx1-ubyte
├── t10k-images-idx3-ubyte/t10k-images.idx3-ubyte
└── t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte
```

### Addestramento

```bash
python main.py
# scegliere: 1
```

- Se esiste `net.json` ne riprende i pesi, altrimenti li genera casualmente.
- Addestra per 10 epoche salvando `net_trained.json` a fine di ogni epoca.
- Al termine esegue automaticamente la valutazione sul test set.

### Inferenza (testing)

```bash
python main.py
# scegliere: 2
```

La fase di testing carica i pesi addestrati da `net_trained.json`, esegue il forward pass (sigmoide + softmax) su tutte le 10.000 immagini del test set MNIST e stampa la percentuale di successo (confronto tra l'`argmax` dell'output e la label vera).

### 📊 Risultati

Accuratezza ottenuta sul test set MNIST (10.000 immagini) dopo 10 epoche di addestramento:

| Metrica | Valore |
|---|---|
| Accuratezza | **96.24 %** |


## 🔍 Dettagli implementativi interessanti

- **Lettura binaria IDX** dei file MNIST con il modulo `struct` (big-endian) e `np.frombuffer`, senza librerie esterne.
- **Softmax numericamente stabile** tramite sottrazione del massimo: `z − max(z)`.
- **One-hot encoding** vettorializzato delle label.
- **Persistenza JSON** dei pesi: la rete può essere salvata e ricaricata tra sessioni (formato `{"layerN": {"w": [...], "b": [...], "a": [...]}}`).
- Tutti i calcoli (sia forward che backward) sono implementati **da zero**, incluso il loop di prodotto pesi-attivazioni, per massima trasparenza didattica.

## 🎯 Possibili miglioramenti

- Vettorizzare il forward pass con `np.dot` (enorme speed-up).
- Sostituire `net.json`/`net_trained.json` con `np.savez` (formato binario più compatto ed efficiente).
- Aggiungere ottimizzatori avanzati (Adam, Momentum), learning rate scheduling, early stopping.
- Aggiungere layer con ReLU + He initialization per confrontare le performance.
