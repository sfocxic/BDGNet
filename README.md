# BDGNet
 A three-branch structure architecture: BDGNet, which parses boundary information, detailed texture information, and global context information, respectively.
# Folder Structure
        ├── dataset_vh_256(vaihingen)
        │   ├── Train
        │   │   ├── Images
        │   │   ├── Labels
        │   ├── Test
        │   │   ├── Images
        │   │   ├── Labels
        ├── potsdam (the same with vaihingen)
        ├── model
        │   ├── BDGNet
        ├── my_dataset.py
        ├── predict.py
        ├── train.py
        ├── transforms.py
        ├── utils.py
# Network Structure
![image](https://github.com/sfocxic/BDGNet/blob/main/fig/Network_Structure.png)
# Requirements：
      numpy~=1.19.5
      torch~=1.11.0
      matplotlib~=3.5.2
      opencv-python~=4.5.5.64
      tqdm~=4.64.0
      Pillow~=9.0.1
      torchvision~=0.12.0
