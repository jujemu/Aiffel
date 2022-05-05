'''
참고자료 링크
https://github.com/WegraLee/deep-learning-from-scratch
'''
import os
import sys
import matplotlib.pylab as plt
import numpy as np

sys.path.append(os.pardir)
from dataset.mnist import load_mnist


# SSE; sum of squares for error
def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# CEE; cross entropy error
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


y = np.array([0.1, 0.05, 0.6, 0.25, 0, 0, 0, 0, 0, 0])
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
print("손실함수 SSE & CEE")
print(sum_squares_error(y, t))
print(cross_entropy_error(y, t))

# loading mnist image_ndarray
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

# mini batch learning
np.random.seed(45)
batch_mask = np.random.choice(x_train.shape[0], size=100)
x_train = x_train[batch_mask]
t_train = t_train[batch_mask]
# print(t_train)


# now, get the machine learn from mini-batch
# if t is one-hot encoding label
def cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    return -np.sum(t * np.log(y + delta)) / y.shape[0]


# t is not one-hot encoding
def cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size


def numerical_diff(f, x):
    h = 1e-4
    fxh1 = f(x+h)
    fxh2 = f(x-h)
    return (fxh1 - fxh2)/2*h


def function_1(x):
    return 0.01*x**2 + 0.1*x


x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
# plt.plot(x, y)
# plt.show()

print("\n수치미분값 계산")
print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))


# partial differential
def function_2(x):
    return x[0]**2 + x[1]**2


# gradient
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] += h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1-fxh2)/2*h
        x[idx] = tmp_val

    return grad


print()
print("gradient를 이용한 수치 편미분계수 구하기")
print(numerical_gradient(function_2, np.array([3.0, 4.0])))
