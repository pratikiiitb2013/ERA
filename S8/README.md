# Assignment
Change the dataset to CIFAR10

Make this network:
C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10

Keep the parameter count less than 50000

Try and add one layer to another

Max Epochs is 20

You are making 3 versions of the above code (in each case achieve above 70% accuracy):

Network with Group Normalization

Network with Layer Normalization

Network with Batch Normalization

Following is link of Notebook
[Notebook](https://github.com/gdeotale/ERA/blob/main/Session8/S8.ipynb)
## Cifar dataset samples
![Images](https://github.com/gdeotale/ERA/assets/8176219/320c6e43-6947-4920-9073-d818adfeb680)
## Here is summary of number of params
Following is link of model [Model](https://github.com/gdeotale/ERA/blob/main/Session8/S8.ipynb)

![Model](https://github.com/gdeotale/ERA/assets/8176219/e62f42e9-c3a5-45e1-b48d-aef23763cf49)

# Layer Norm
### Here is output of final few epochs
![layer_Capture](https://github.com/gdeotale/ERA/assets/8176219/7538d3b9-5332-4fc3-b87d-f6dde38abc12)
### Corresponding output curve
![layer_curve](https://github.com/gdeotale/ERA/assets/8176219/aced2555-e483-4ff5-8adb-40c7626a20f3)
### Missed Images
![layer_miss](https://github.com/gdeotale/ERA/assets/8176219/b80226f4-a157-4773-95cb-151094888dea)

# Batch Norm
### Here is output of final few epochs
![BN_capture](https://github.com/gdeotale/ERA/assets/8176219/879e076d-a2bb-4a31-87a6-d6fbf02f9503)
### Corresponding output curve
![BN_curve](https://github.com/gdeotale/ERA/assets/8176219/7c6e9aa1-95d2-4df1-804b-0ccc0458eaab)
### Missed Images
![bn_miss](https://github.com/gdeotale/ERA/assets/8176219/9a25409c-b5ec-422d-bc6c-43c8d7a3a3bd)

# Group Norm
### Here is output of final few epochs
![group_Capture](https://github.com/gdeotale/ERA/assets/8176219/8fcc1dc9-17c3-42e2-a986-e824d8481134)
### Corresponding output curve
![group_curve](https://github.com/gdeotale/ERA/assets/8176219/595f4ada-bd2d-430c-baf4-a62f0d0a2b93)
### Missed Images
![group_miss](https://github.com/gdeotale/ERA/assets/8176219/e5297fb2-613e-45d4-9b04-c695ef3de86d)
