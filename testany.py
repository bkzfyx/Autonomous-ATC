import visdom
from visdom import Visdom
import numpy as np
import time
vis = visdom.Visdom()
#vis.text('Hello, world!')
#vis.image(np.ones((3, 10, 10)))
#多图像显示与更新demo
opts1 = {
        "title": 'train_data',
        "xlabel": 'x',
        "ylabel": 'y',
        "width": 600,
        "height": 400,
        "legend": ['goals_made','collisions_made']
}
opts2 = {
        "title": 'train_data2',
        "xlabel": 'id',
        "ylabel": 'speed',
        "width": 600,
        "height": 400,
        "numbins":30
}
wind = Visdom()
wind2 = Visdom()
        # 初始化窗口信息
wind.line(X=[0.], # Y的第一个点的坐标
Y=[[0.,0.]], # X的第一个点的坐标
win = 'train_data', # 窗口的名称
opts = opts1 # 图像的标例
)
wind2.bar(X=[251,0,0,0,0,0,0,0,0,0,0,0,0,0,0,251,0,0,0,0,0,0,251,0,0,0,0,0,0,0],win = 'train_data2',opts=opts2)
