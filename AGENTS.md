该项目是一个基于yolov13模型进行改进的项目，用来进行水下光学数据集URPC的检测，目的是通过改进使模型能够提升检测的精度，包括P、R、mAP、APS、APM、APL各项指标。
最核心的目标是为了有足够的创新点来支撑SCI论文。
项目对应的服务器目录为/home/room305/ZZF/yolov13yuan-6000
其中train.py是训练文件,test.py是测试文件，data.yaml是数据集的配置文件。
ultralytics\cfg\models\v13\yolov13.yaml 这是该项目模型的结构配置文件。
ultralytics\nn\modules\block.py 这是该项目里相关模块的定义代码保存文件。
ultralytics\utils\loss.py 这是模型的损失函数文件。
ultralytics\nn\modules\__init__.py 这是模型各模块的注册文件。
原始yolov13的结构配置如下所示：
nc: 3 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov13n.yaml' will call yolov13.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]   # Nano
  s: [0.50, 0.50, 1024]   # Small
  l: [1.00, 1.00, 512]    # Large
  x: [1.00, 1.50, 512]    # Extra Large

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv,  [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]] # 1-P2/4
  - [-1, 2, DSC3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]] # 3-P3/8
  - [-1, 2, DSC3k2,  [512, False, 0.25]]
  - [-1, 1, DSConv,  [512, 3, 2]] # 5-P4/16
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, DSConv,  [1024, 3, 2]] # 7-P5/32
  - [-1, 4, A2C2f, [1024, True, 1]] # 8

head:
  - [[4, 6, 8], 2, HyperACE, [512, 8, True, True, 0.5, 1, "both"]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [ 9, 1, DownsampleConv, []]
  - [[6, 9], 1, FullPAD_Tunnel, []]  #12     
  - [[4, 10], 1, FullPAD_Tunnel, []]  #13    
  - [[8, 11], 1, FullPAD_Tunnel, []] #14 
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 12], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, DSC3k2, [512, True]] # 17
  - [[-1, 9], 1, FullPAD_Tunnel, []]  #18

  - [17, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 13], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, DSC3k2, [256, True]] # 21
  - [10, 1, Conv, [256, 1, 1]]
  - [[21, 22], 1, FullPAD_Tunnel, []]  #23
  
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 18], 1, Concat, [1]] # cat head P4
  - [-1, 2, DSC3k2, [512, True]] # 26
  - [[-1, 9], 1, FullPAD_Tunnel, []]  

  - [26, 1, Conv, [512, 3, 2]]
  - [[-1, 14], 1, Concat, [1]] # cat head P5
  - [-1, 2, DSC3k2, [1024,True]] # 30 (P5/32-large)
  - [[-1, 11], 1, FullPAD_Tunnel, []]  
  
  - [[23, 27, 31], 1, Detect, [nc]] # Detect(P3, P4, P5)
