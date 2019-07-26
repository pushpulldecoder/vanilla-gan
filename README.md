# Vanilla Gan

> Looking down the misty path to uncertain destinations🌌🍀&nbsp;&nbsp;&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- x' <br><br><br>

It is implementation of Vanilla GAN using PyTorch over GPU according to paper by Ian Goodfellow et al., 2014 available at<br>
https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf <br>
<br>

## Discriminator Network
<pre>




                     [no_data, gan_out_size]

                                ||
                                ||
                                ||
                               \||/
                                \/
                       ____________________
                      |                    |
                      |   LeakyRelu(0.3)   |
            hidden0   |                    |   [no_data, gan_out_size] X [gan_out_size, 512]  ->  [no_data, 512]
                      |    Dropout(0.2)    |
                      |                    |
                      |____________________|

                                ||
                                ||
                          [no_data, 512]
                                ||
                               \||/
                                \/
                       ____________________
                      |                    |
                      |   LeakyRelu(0.3)   |
            hidden1   |                    |   [no_data, 512] X [512, 256]  ->  [no_data, 256]
                      |    Dropout(0.2)    |
                      |                    |
                      |____________________|

                                ||
                                ||
                          [no_data, 256]
                                ||
                               \||/
                                \/
                       ____________________
                      |                    |
                      |   LeakyRelu(0.3)   |
            hidden2   |                    |   [no_data, 256] X [256, 128]  ->  [no_data, 128]
                      |    Dropout(0.2)    |
                      |                    |
                      |____________________|

                                ||
                                ||
                          [no_data, 128]
                                ||
                               \||/
                                \/
                       ____________________
                      |                    |
                      |                    |
            out       |      Sigmoid       |   [no_data, 512] X [128, 1]  ->  [no_data, 1]
                      |                    |
                      |                    |
                      |____________________|

                                ||
                                ||
                                ||
                               \||/
                                \/

                           [no_data, 1]

                     Probability of Data entered
                   sampled from same distribution
                      


</pre>
<br>
<br>
<pre>




                       [no_data, gan_in_size]

                                ||
                                ||
                                ||
                               \||/
                                \/
                       ____________________
                      |                    |
                      |                    |
            hidden0   |   LeakyRelu(0.3)   |   [no_data, gan_in_size] X [gan_in_size, 128]  ->  [no_data, 128]
                      |                    |
                      |                    |
                      |____________________|

                                ||
                                ||
                          [no_data, 512]
                                ||
                               \||/
                                \/
                       ____________________
                      |                    |
                      |                    |
            hidden1   |   LeakyRelu(0.3)   |   [no_data, 128] X [128, 256]  ->  [no_data, 256]
                      |                    |
                      |                    |
                      |____________________|

                                ||
                                ||
                          [no_data, 256]
                                ||
                               \||/
                                \/
                       ____________________
                      |                    |
                      |                    |
            hidden2   |   LeakyRelu(0.3)   |   [no_data, 256] X [256, 512]  ->  [no_data, 512]
                      |                    |
                      |                    |
                      |____________________|

                                ||
                                ||
                          [no_data, 128]
                                ||
                               \||/
                                \/
                       ____________________
                      |                    |
                      |                    |
            out       |      Sigmoid       |   [no_data, 512] X [512, gan_out_size]  ->  [no_data, gan_out_size]
                      |                    |
                      |                    |
                      |____________________|

                                ||
                                ||
                                ||
                               \||/
                                \/

                      [no_data, gan_out_size]

                     Probability of Data entered
                   sampled from same distribution
                      


</pre>
