# AutoEncoding Neural Networks on MNIST

## Authors : 
 - Mehdi Badreddine Ghouwati
 - Youssef Achenchabe
## Our work
The main purpose of our work is to get familiar with the AutoEncoding Neural Networks and how to implement it using PyTorch. The notebooks are inspired from the MNIST examples presented [here](https://github.com/L1aoXingyu/pytorch-beginner/tree/master/08-AutoEncoder)

## The dataset
We worked with the MNIST dataset.


## AutoEncoder Implementation on MNIST
We considered two models :
 ### A simple one :
 * Fully Connected Layers (784,400) and (400,20) with a ReLu activation fuction,
 * The Loss Function considered is a BCE,
 * The Encoder and the Decoder are symmetric.
```python
class simple_ae(nn.Module):
    def __init__(self):
        super(simple_ae, self).__init__()
        # encoder :FC (784 -> 400), RELU activation; FC (400, 20)
        self.encoder = nn.Sequential(
                nn.Linear(784, 400),
                nn.ReLU(),
                nn.Linear(400, 20),
                )
        # decoder :FC (20 -> 400), RELU activation; FC (400, 784)
        self.decoder = nn.Sequential(
                nn.Linear(20, 400),
                nn.ReLU(),
                nn.Linear(400, 784),
                )
```

 ### A more elaborated one :
 * Fully Connected Layers (784,128), (128,128), (128,16) and (16,4) with a ReLu activation fuction,
 * The Loss Function considered is a MSE,
 * The Encoder and the Decoder are symmetric.
 ```python
 class elab_ae(nn.Module):
    def __init__(self):
        super(elab_ae, self).__init__()
        # create using nn.Sequential()
        # encoder :FC (784 -> 400), RELU activation; FC (400, 20)
        # decoder :FC (20 -> 400), RELU activation; FC (400, 784)
        self.encoder = nn.Sequential(
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128,16),
                nn.ReLU(),
                nn.Linear(16,4)
                )
        self.decoder = nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(),
                nn.Linear(16, 128),
                nn.ReLU(),
                nn.Linear(128,128),
                nn.ReLU(),
                nn.Linear(128,784)
                )
 ```

### Results

#### Model 1 :

![ReconstructionONE](https://github.com/mbghouwa/AE-AutoEncoder_PyTorch/tree/master/images/ae_1_recon.png)


![GenerationONE](https://github.com/mbghouwa/AE-AutoEncoder_PyTorch/tree/master/images/ae_1_gene.png)

#### Model 2 :

![ReconstructionTWO](https://github.com/mbghouwa/AE-AutoEncoder_PyTorch/tree/master/images/ae_2_recon.png)

![GenerationTWO](https://github.com/mbghouwa/AE-AutoEncoder_PyTorch/tree/master/images/ae_2_gene.png)

## AutoEncoder Implementation on MNIST

The encoder we considered is a three-layered Fully Conneted Network :

* FC(784,400) and 2 FC (400,20) to define the mu and var
* The loss function considered is a mix between the BCE and the KLD
* The decoder isn't symmetric in this case and is a FC Network with one hidden layer, applied on the reparametrized mu and var.

```python
class vae(nn.Module):
    def __init__(self):
        
        super(vae, self).__init__()
        # encoding_layers :FC (784 -> 400), 2 FC (mu and var) (400 -> 20)
        self.fc1=nn.Linear(784,400)
        self.fcmu=nn.Linear(400,20)
        self.fcvar=nn.Linear(400,20)
        
        # decoder :FC (20 -> 400), RELU activation; FC (400, 784)
        self.decoder = nn.Sequential(
                nn.Linear(20, 400),
                nn.ReLU(),
                nn.Linear(400, 784),
                )
```
