# AAE
Adversarial Auto Encoder (AAE) with tensorflow for MNIST

# usage
- training
```
python main.py -d [MNIST csv file]
```
- making figures after training
```
python using_result.py -d [MNIST csv file]
```

- MNIST csv files can be made by following script.
https://pjreddie.com/projects/mnist-in-csv/

# 100 epoch results
![AAE](results/AAE.png)

## Random sample from each Gaussian

### 0
![0](results/0.png)

### 1
![1](results/1.png)

### 2
![2](results/2.png)

## walking from (0, 0) to (2, 0)
![0to1](results/0to1.png)

## walking along with constant radius
![const_r](results/const_r.png)