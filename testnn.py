# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:24:47 2018

@author: ccc
"""
import mnist_loader
training_data, validation_data, test_data = \
    mnist_loader.load_data_wrapper()
import network
net = network.Network([784, 50, 10])

net.SGD(training_data, 3, 10, 3.0, test_data=test_data)

