# Adversarial Training and Visualization on CIFAR-10


## Update

* (2019/4/18) Change the default alpha from 2 to 2/255, and update the results.

## Results

### Learning Curves

Epsilon in linf (l2) training is 0.0157 (0.314). [0.0157=4/255, 0.314=80/255]

<table border=0 width="50px" >
    <tbody> 
    <tr>    
        <th colspan="2" align="center"> <strong>Standard Training</strong> </th>
        <th colspan="2" align="center"> <strong>l_inf Training</strong> </th>
        <th colspan="2" align="center"> <strong>l_2 Training</strong></th>
    </tr>
    <tr>
        <th colspan="2" align="center"> <img src="https://github.com/louis2889184/adversarial_training/blob/master/cifar-10/img/cifar_learning_curve_std.jpg"> </th>
        <th colspan="2" align="center"> <img src="https://github.com/louis2889184/adversarial_training/blob/master/cifar-10/img/cifar_learning_curve_linf.jpg"> </th>
        <th colspan="2" align="center"> <img src="https://github.com/louis2889184/adversarial_training/blob/master/cifar-10/img/cifar_learning_curve_l2.jpg"> </th>
    </tr>
    <tr>
        <th colspan="1" align="center"> <strong>Standard Accuracy</strong> <br/> (train/test) </th>
        <th colspan="1" align="center"> <strong>Robustness Accuracy</strong> <br/> (train/test) </th>
        <th colspan="1" align="center"> <strong>Standard Accuracy</strong> <br/> (train/test) </th>
        <th colspan="1" align="center"> <strong>Robustness Accuracy</strong> <br/> (train/test) </th>
        <th colspan="1" align="center"> <strong>Standard Accuracy</strong> <br/> (train/test) </th>
        <th colspan="1" align="center"> <strong>Robustness Accuracy</strong> <br/> (train/test) </th>
    </tr>
    <tr>
        <th colspan="1" align="center"> 92.19/87.14 </th>
        <th colspan="1" align="center"> 0.00/7.85 </th>
        <th colspan="1" align="center"> 79.69/78.09 </th>
        <th colspan="1" align="center"> 61.72/63.8 </th>
        <th colspan="1" align="center"> 89.84/85.39 </th>
        <th colspan="1" align="center"> 76.56/77.76 </th>
    </tr>
    </tbody>
</table>

Note that in testing mode, the target label used in creating the adversarial example is the most confident prediction of the model, not the ground truth. Therefore, sometimes the testing robustness is higher than training robustness, when the prediction is wrong at first. <br/>

Learning rate is manually changed during training: <br/>

* `0.1` in epoch `[0, 100]`
* `0.01` in epoch `[100, 150]`
* `0.001` in epoch `[150, 200]`

the policy is followed https://github.com/MadryLab/cifar10_challenge.

### Visualization of Gradient with Respect to Input

![visualization](https://github.com/louis2889184/adversarial_training/blob/master/cifar-10/img/cifar_grad_default.jpg)

### The Adversarial Example with large epsilon

The maximum epsilon is set to 4.7 (l2 norm) in this part.

![large](https://github.com/louis2889184/adversarial_training/blob/master/cifar-10/img/cifar_large_l2_default.jpg)


## Requirements:
```
python >= 3.5
torch == 1.0
torchvision == 0.2.1
numpy >= 1.16.1
matplotlib >= 3.0.2
```

## Execution

### Training

Standard training: <br/>

```
python src/main.py --data_root [data directory]
```

linf training: <br/>

```
python src/main.py --data_root [data directory] -e 0.0157 -p 'linf' --adv_train
```

l2 training: <br/>

```
python src/main.py --data_root [data directory] -e 0.314 -p 'l2' --adv_train
```

### Visualization

visualize gradient to input: <br/>

```
python src/visualize.py --load_checkpoint [your_model.pth]
```

visualize adversarial examples with larger epsilon <br/>

```
python src/visualize_attack.py --load_checkpoint [your_model.pth]
```

## Training Time

Standard training: 78 s / 100 iterations <br/>
Adversarial training: 784 s / 100 iterations <br/> <br/>

where the batch size is 128 and train on NVIDIA GeForce GTX 1080.
