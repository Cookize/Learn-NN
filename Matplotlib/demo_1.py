import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# fig = plt.figure()
# fig.suptitle('No axes on this figure')  # Add a title so we know which it is
# fig, ax_lst = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes
# plt.show()

# x = np.linspace(0, 2, 100)
# plt.plot(x, x, label='linear')
# plt.plot(x, x**2, label='quadratic')
# plt.plot(x, x**3, label='cubic')
# plt.xlabel('x label')
# plt.ylabel('y label')
# plt.legend()
# plt.show()

# def my_plotter(ax, data1, data2, param_dict):
#     """
#     A helper function to make a graph
#
#     Parameters
#     ----------
#     ax : Axes
#         The axes to draw to
#
#     data1 : array
#        The x data
#
#     data2 : array
#        The y data
#
#     param_dict : dict
#        Dictionary of kwargs to pass to ax.plot
#
#     Returns
#     -------
#     out : list
#         list of artists added
#     """
#     out = ax.plot(data1, data2, **param_dict)
#     return out
#
#
# data1, data2, data3, data4 = np.random.randn(4, 100)
# fig, (ax1, ax2) = plt.subplots(2, 1)
# my_plotter(ax1, data1, data2, {'marker': 'x'})
# my_plotter(ax2, data3, data4, {'marker': 'o'})

# plt.ioff()
# # for i in range(3):
# #     plt.plot(np.random.rand(10))
# # plt.show()

# t = np.arange(0., 5., 0.2)
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
# plt.show()

data = {
    'a': np.arange(50),
    'c': np.random.randint(0, 50, 50),
    'd': np.random.randn(50),
}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

plt.scatter('a', 'b', c = 'c', s = 'd', data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.show()