# coding=utf-8
'''
Created on 2015-10-15
@author: kwsy2015
'''
import zmq

context = zmq.Context()
print('connect to hello world server')

socket = context.socket(zmq.REQ)
socket.connect('tcp://192.168.3.6:5555')

for request in range(1, 10):
    print('send ', request, '...')
    socket.send_string('hello')
    message = socket.recv()
    print('received reply ', request, '[', message, ']')