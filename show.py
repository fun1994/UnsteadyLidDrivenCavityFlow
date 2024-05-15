# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:27:26 2024

@author: HFKJ059
"""

import numpy as np
from matplotlib import pyplot as plt


def read_1d(path, filename):
    with open("./data/" + path + "/" + filename + ".txt", "r") as file:
        data = file.read()
    data = data.split()
    for i in range(len(data)):
        data[i] = float(data[i])
    data = np.array(data)
    return data

def read_3d(path, filename, Nx, Ny):
    data = []
    with open("./data/" + path + "/" + filename + ".txt", "r") as file:
        while True:
            line = file.readline()
            if not line:
                break
            data_temp = line.split()
            data.append([])
            for i in range(Nx):
                data[-1].append([])
                for j in range(Ny):
                    data[-1][-1].append(data_temp[i * Ny + j])
    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                data[i][j][k] = float(data[i][j][k])
    data = np.array(data)
    return data

def read(grid):
    x_p = read_1d(grid, "x_p")
    y_p = read_1d(grid, "y_p")
    x_u = read_1d(grid, "x_u")
    y_u = read_1d(grid, "y_u")
    x_v = read_1d(grid, "x_v")
    y_v = read_1d(grid, "y_v")
    t = read_1d(grid, "t")
    p = read_3d(grid, "p", x_p.shape[0], y_p.shape[0])
    u = read_3d(grid, "u", x_u.shape[0], y_u.shape[0])
    v = read_3d(grid, "v", x_v.shape[0], y_v.shape[0])
    return x_p, y_p, x_u, y_u, x_v, y_v, t, p, u, v

def show(x, y, u, v, grid, time):
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots()
    ax.streamplot(X, Y, u.T, v.T, density=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("grid=" + grid + ", time=" + time)
    plt.xlim(0, (3 * x[-1] - x[-2]) / 2)
    plt.ylim(0, (3 * y[-1] - y[-2]) / 2)
    ax.set_aspect("equal")
    plt.show()

def run(grid):
    x_p, y_p, x_u, y_u, x_v, y_v, t, p, u, v = read(grid)
    period = 250
    for i in range(1, 21):
        if grid == "staggered":
            u_temp = (u[i * period, :-1, :] + u[i * period, 1:, :]) / 2
            v_temp = (v[i * period, :, :-1] + v[i * period, :, 1:]) / 2
        elif grid == "collocated":
            u_temp = u[i * period, :, :]
            v_temp = v[i * period, :, :]
        show(x_v, y_u, u_temp, v_temp, grid, str(t[i * period]))

def main():
    run("staggered")
    run("collocated")


main()
