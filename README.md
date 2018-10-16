# IDCN
Implicit Dual-domain Convolutional Network for Robust Color Image Compression Artifact Reduction

## Network Architecture
### Architecture
![Error](https://github.com/zhenngbolun/IDCN/blob/master/Fig4.png)

### Dual-domain Correction Unit
![Error](https://github.com/zhenngbolun/IDCN/blob/master/Fig6.png)

## Results
### LIVE1 Result
QF | DMCNN | S-Net(L8) | IDCN |            
----- |:-------------:|:-------------:|:-------------:|  
 20    | 32.09/0.905  | 32.26/0.9067  | 32.32/0.9077  
 10    | 29.73/0.842  | 29.87/0.8467  | 29.99/0.8503
 
### BSDS500 Result
QF | DMCNN | S-Net(L8) | IDCN |            
----- |:-------------:|:-------------:|:-------------:|  
 20    | 31.98/0.904  | 32.15/0.9047  | 32.22/0.9058  
 10    | 29.67/0.840  | 29.82/0.8440  | 29.94/0.8475 
 
### WIN143 Result
QF | ARCNN-Color | S-Net(L8) | IDCN |            
----- |:-------------:|:-------------:|:-------------:|  
 20    | 34.08/0.9179  | 34.61/0.9250  | 34.74/0.9263  
 10    | 31.76/0.8707  | 32.15/0.8795  | 32.46/0.8847

![Error](https://github.com/zhenngbolun/IDCN/blob/master/Fig7.png)
![Error](https://github.com/zhenngbolun/IDCN/blob/master/Fig9.png)
 
