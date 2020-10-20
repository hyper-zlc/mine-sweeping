# 游戏配置参数

# bg_region:游戏所在的屏幕位置(x,y,w,h)，x,y：区域左上角坐标，w,h：区域的宽和高 ，可通过截图+windows的画图软件获取
# 如果习惯用(x1,y1,x2,y2)的方式表示屏幕上的区域，请输入(x1,y1,x2-x1,y2-y1)
# BG_REGION = (473, 420, 1433 - 473, 932 - 420)
BG_REGION = (722,262, 1202-722,517-262)


# 高级难度的扫雷有16行，30列
ROWS = 16
COLS = 30

# # 计算出每个格子的大小
row_size = BG_REGION[3] / ROWS
cols_size = BG_REGION[2] / COLS

# 编码和解码神经网络的输出。本程序中，-1表示这个格子是旗子，-2表示未翻开的格子，9表示翻开的地雷，0-8表示周围八个格子地雷数量，
net_encoder = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 9: 8, -1: 9, -2: 10}
net_decoder = {v: k for k, v in net_encoder.items()}


# 图片保存路径
PATH = r'C:\Users\Administrator\Desktop\zlc\扫雷\imgs'
