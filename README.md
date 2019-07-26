# Vanilla Gan

> Looking down the misty path to uncertain destinations🌌🍀&nbsp;&nbsp;&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- x' <br><br><br>

It is implementation of Vanilla GAN using PyTorch over GPU according to paper by Ian Goodfellow et al., 2014 available at<br>
https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf <br>
<br>

## GAN Epochs

<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/gan.gif height="300" width="300">

<br>
<br>

## Discriminator Network
<pre>

            [no_data, gan_out_size]

                       ||
                      \||/
                       \/
              ____________________
             |   LeakyRelu(0.3)   |
   hidden0   |                    |   [no_data, gan_out_size] X [gan_out_size, 512] -> [no_data, 512]
             |    Dropout(0.2)    |
             |____________________|

                       ||
                 [no_data, 512]
                      \||/
                       \/
              ____________________
             |   LeakyRelu(0.3)   |
   hidden1   |                    |   [no_data, 512] X [512, 256] ->  [no_data, 256]
             |    Dropout(0.2)    |
             |____________________|

                       ||
                 [no_data, 256]
                      \||/
                       \/
              ____________________
             |   LeakyRelu(0.3)   |
   hidden2   |                    |   [no_data, 256] X [256, 128] -> [no_data, 128]
             |    Dropout(0.2)    |
             |____________________|

                       ||
                 [no_data, 128]
                      \||/
                       \/
              ____________________
             |                    |
   out       |      Sigmoid       |   [no_data, 512] X [128, 1] -> [no_data, 1]
             |                    |
             |____________________|

                       ||
                      \||/
                       \/

                  [no_data, 1]

            Probability of Data entered
          sampled from same distribution
                      


</pre>
<br>
<br>

## Generator Network
<pre>

              [no_data, gan_in_size]

                       ||
                      \||/
                       \/
              ____________________
             |                    |
   hidden0   |   LeakyRelu(0.3)   |   [no_data, gan_in_size] X [gan_in_size, 128] -> [no_data, 128]
             |____________________|

                       ||
                 [no_data, 512]
                      \||/
                       \/
              ____________________
             |                    |
   hidden1   |   LeakyRelu(0.3)   |   [no_data, 128] X [128, 256] -> [no_data, 256]
             |____________________|

                       ||
                 [no_data, 256]
                      \||/
                       \/
              ____________________
             |                    |
   hidden2   |   LeakyRelu(0.3)   |   [no_data, 256] X [256, 512] -> [no_data, 512]
             |____________________|

                       ||
                 [no_data, 128]
                       ||
                      \||/
                       \/
              ____________________
             |                    |
   out       |      Sigmoid       |   [no_data, 512] X [512, gan_out_size] -> [no_data, gan_out_size]
             |____________________|

                       ||
                      \||/
                       \/

              [no_data, gan_out_size]

            Probability of Data entered
          sampled from same distribution
           
</pre>
<br>
<br>

## Dependencies
* Torch
    * https://pytorch.org/
* Torchvision
    * https://pytorch.org/
* Numpy
    * pip install numpy
* Pandas
    * pip install pandas
* Matplotlib
    * pip install matplotlib
* tensorboardX
    * pip install tensorboardX
* IPython
    * pip install ipython

<br>
<br>

## Loading Dataset
PyTorch dataset loader will load MNIST dataset from <i>HOME_FOLDER</i> and if not present, it will download <br>
Change it as required to load other database
```python
DATA_FOLDER = '/home/pushpull/mount/intHdd/dataset/'

def load_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5,), (.5,))
        ])
    out_dir = '{}/'.format(DATA_FOLDER)
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

data = load_data()
```

## Discriminator Network
Discriminator network is extension of <b><i>torch.nn.Module</i></b> class already containing sequence models <br>
Creating <i>discriminator</i> class object will initialize class with fully connected network with 3 hidden layers and one output layer <br>
It gives probability of input being sampled from same distribution as that of training data <br>
```python
def __init__(self, parameter):
    super(discriminator, self).__init__()
    self.y = 1
    self.gan_out = parameter.get("gan_out")
    self.Network()
```
Initialization of hidden layers <br>
```python
def Network(self):
            self.hidden0 = nn.Sequential( 
                nn.Linear(self.gan_out, 512),
                nn.LeakyReLU(0.3),
                nn.Dropout(0.2)
                ).cuda()
            self.hidden1 = nn.Sequential(
                nn.Linear(512, 256),
                nn.LeakyReLU(0.3),
                nn.Dropout(0.2)
                ).cuda()
            self.hidden2 = nn.Sequential(
                nn.Linear(256, 128),
                nn.LeakyReLU(0.3),
                nn.Dropout(0.2)
                ).cuda()
            self.out = nn.Sequential(
                torch.nn.Linear(128, self.y),
                torch.nn.Sigmoid()
            ).cuda()
```
Feedforward implementation <br>
```python            
def forward(self, x_):
        x_ = self.hidden0(x_)
        x_ = self.hidden1(x_)
        x_ = self.hidden2(x_)
        x_ = self.out(x_)
        return x_
```

## Generator Network
Generator network is extension of <b><i>torch.nn.Module</i></b> class already containing sequence models <br>
Creating <i>discriminator</i> class object will initialize class with fully connected network with 3 hidden layers and one output layer <br>
```python
def __init__(self, parameter):
    super(generator, self).__init__()
    self.gan_out = parameter.get("gan_out")
    self.gan_in  = parameter.get("gan_in")
    self.Network()
```
Initialization of hidden layers <br>
```python
def Network(self):
        self.hidden0 = nn.Sequential(
            nn.Linear(self.gan_in, 128),
            nn.LeakyReLU(0.3)
        ).cuda()
        self.hidden1 = nn.Sequential(            
            nn.Linear(128, 256),
            nn.LeakyReLU(0.3)
        ).cuda()
        self.hidden2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.3)
        ).cuda()
    
        self.out = nn.Sequential(
            nn.Linear(512, self.gan_out),
            nn.Tanh()
        ).cuda()
```
Feedforward implementation <br>
```python
def forward(self, x_):
        x_ = self.hidden0(x_)
        x_ = self.hidden1(x_)
        x_ = self.hidden2(x_)
        x_ = self.out(x_)
        return x_
```

## Noise Generator
Noise generator will return random values sampled uniformly from normal distribution <br>
```python
def noise(length, size):
    noise = Variable(torch.randn(length, size)).cuda()
    return noise
```

## Train GAN
Here, <i>discriminator</i> network and <i>generator</i> network both are being trained together <br>
For every epoch, first <i>discriminator</i> network is being trained and next <i>generator</i> network <br>
Discriminator is trained for both, true data and random noise <br>

### To use CPU only, remove <i>cuda()</i> from everywhere

## Generated Images
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step0.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step10.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step20.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step30.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step40.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step50.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step70.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step90.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step170.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step200.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step790.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step2330.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step2550.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step2870.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step3150.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step3210.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step3670.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step3910.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step4230.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step4240.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step4530.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step5320.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step5550.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step5890.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step6000.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step6760.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step6980.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step7640.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step7750.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step8190.png height="300" width="300">
<img src=https://github.com/pushpull13/vanilla-gan/blob/master/img/Step10900.png height="300" width="300">
