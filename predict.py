import os
import math
import argparse
import datetime
import time

import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import torch.nn as nn
import PIL.Image as Image
from torch.utils.tensorboard import SummaryWriter
import transforms as T
from my_dataset import MyDataSet, VOCSegmentation, PotsdamSegmentation
from model.BDGnet.BDGnet import BDGnet
from torch.backends import cudnn

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.cuda.manual_seed(0)
cudnn.enabled = True
cudnn.benchmark = True



def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./seg_weights/{}".format(args.save_path)) is False:
        os.makedirs("./seg_weights/{}".format(args.save_path))


    data_transform = {
        "test": T.Compose([T.Resize(256),
                                   T.CenterCrop(256),
                                   T.ToTensor(),
                                   T.Normalize([0.3412, 0.3637, 0.3378], [0.1402, 0.1384, 0.1439])])}

    # 实例化Potsdam训练数据
    test_dataset = PotsdamSegmentation(args.data_path,
                                  transforms=data_transform['test'],
                                  txt_name="test",
                                  predict=True)
    print("Data initialization is finish!")


    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers


    model = BDGnet(n_classes=6).to(device)

    #输出模型参数量
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    # 输出模型所有key值
    # for key, value in model.named_parameters():
    #     print(key)
    interp = nn.Upsample(size=(256, 256), mode='bilinear')

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    start_time = time.time()

    model_name="BDGNet"
    data_name="pot"
    if os.path.exists("./{}/{}".format(data_name,model_name)) is False:
        os.makedirs("./{}/{}".format(data_name,model_name))
    print('Testing.......')
    model.eval()
    for index,batch in enumerate(test_dataset):
        image,filename,_=batch
        image=torch.unsqueeze(image,dim=0)
        name='mask_'+filename.split('/')[-1].replace('jpg', 'png')
        image=image.float().to(device)
        with torch.no_grad():
            pred=model(image)
        pred=pred['out']
        pred=pred.argmax(1).squeeze(0).squeeze(0)
        pred = pred.to("cpu").numpy().astype(np.uint8)
        img=Image.fromarray(pred,'P')
        colormap=[0,0,255]+[0, 255, 255]+[0, 255, 0]+ \
                 [255, 255, 0]+[255,0,0]+[255, 255, 255]
        img.putpalette(colormap)
        img.save("./"+data_name+"/"+model_name+"/"+name)

    total_time = time.time() - start_time
    print("test time {}".format(total_time))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--data-path', type=str,
                        default="./dataset_256")
    parser.add_argument('--model-name', default='', help='create model name')
    parser.add_argument("--aux", default=False, type=bool, help="auxilier loss")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:2', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--save-path', default="BDGNet", help='The path to save the loss graph')
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    opt = parser.parse_args()

    main(opt)

