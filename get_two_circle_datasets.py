from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from ripser import Rips
import math

N = 200
N_per_class = int(N / 2)
N_in_class = 200

def noise(N, scale):
    return scale * np.random.random((N, 2))

def circle(N, scale, offset):
    return offset + scale * datasets.make_circles(n_samples=N, factor=0.5, noise=0.02)[0]#[1] represents labels

def circle_no_noise(N, scale, offset):
    return offset + scale * datasets.make_circles(n_samples=N, factor=0.5, noise=0)[0]#[1] represents labels



def ellipse_point(a, b, center, num_points):
    # 参数a和b定义椭圆的长轴和短轴，center是椭圆的中心
    t = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + a * np.cos(t) * np.sqrt(1 - (b / a) ** 2 * np.sin(t) ** 2)
    y = center[1] + b * np.sin(t) * np.sqrt(1 - (b / a) ** 2 * np.cos(t) ** 2)
    return x, y


def random_point_on_ellipse(a, b, center, num_points):
    # 在椭圆上随机取点
    t = np.random.uniform(0, 2 * np.pi, num_points)
    x = center[0] + a * np.cos(t) * np.sqrt(1 - (b / a) ** 2 * np.sin(t) ** 2)
    y = center[1] + b * np.sin(t) * np.sqrt(1 - (b / a) ** 2 * np.cos(t) ** 2)
    return x, y

def noise_Gaussian(points, mean):
    noise = np.random.normal(0,mean,points.shape)
    out = points+noise
    return out


def get_random_two_ellipse(a,b,a_s,b_s,center,num_points):
    random_x, random_y = random_point_on_ellipse(a, b, center, num_points)
    random_xs, random_ys = random_point_on_ellipse(a_s, b_s, center, num_points)
    random_x = np.append(random_x,random_xs)
    random_y = np.append(random_y, random_ys)
    random_data = np.c_[random_x,random_y]
    random_data = noise_Gaussian(random_data,0.5)
    return random_data

def random_point_on_circle(radius, center,num_points):
    # 生成一个在[0, 2π)范围内的随机角度
    angle = np.random.uniform(0, 2 * math.pi, num_points)

    # 计算点的坐标
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)

    return x, y

def get_random_small_circle_add_noise(radius, center, num_points):
    random_x, random_y = random_point_on_circle(radius, center,num_points)
    random_data = np.c_[random_x, random_y]
    random_data = noise_Gaussian(random_data, 0.5)
    #print(random_data.shape) (100,2)
    return random_data

def get_random_small_ellipse_add_noise(a, b, center, num_points):
    random_x, random_y = random_point_on_ellipse(a, b, center, num_points)
    random_data = np.c_[random_x, random_y]
    random_data = noise_Gaussian(random_data, 0.5)
    return random_data


"""
适用于ripser,diag形式为list,[array([[0,b1],[0,b2],[0,inf]]),array([[a4,b4]])]
不同Hi分别去掉inf
"""
def is_inf(x):
   return x[1]!=np.inf

def rips_delete_inf(datas,i):
    rips = Rips(maxdim=1, coeff=2)
    diagrams_delete_inf = []
    for data in datas:
        diagram = rips.fit_transform(data)[i]
        diagram_delete_inf_onedata = list(filter(is_inf, diagram))
        #print(diagram_delete_inf_onedata)
        diagrams_delete_inf.append(diagram_delete_inf_onedata)
    return diagrams_delete_inf





