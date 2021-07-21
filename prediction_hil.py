import zmq
from ppo import PPO
import os
import gym
import numpy as np

def predict():
    action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32)
    input_shape = [67]
    model = PPO(input_shape, action_space,
                model_dir=os.path.join("models", 'pretrained_agent'))
    model.init_session(init_logging=False)
    model.load_latest_checkpoint()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://*:5555')

    while True:
        state = socket.recv()
        action = model.predict(state, greedy=True)
        socket.send(action)