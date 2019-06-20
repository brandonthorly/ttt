import os
import shutil

# importing Flask
from flask import Flask, jsonify, render_template, request

#import json
# import tic tac toe game 
from deep_reinforcement_learning import *

import numpy as np
import tensorflow as tf


def setup_session():
    global sess
    global x
    global prediction
    global s
    global graph

    try:
        sess.close()
    except NameError:
        pass

    # setting up session
    sess = tf.InteractiveSession()

    #prediction = neural_network_model(x)

    x , prediction, _ = createNetwork()

    #prediction = convolutional_neural_network(x)
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("model")
    if checkpoint and checkpoint.model_checkpoint_path:
        s = saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded the model:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    graph = tf.get_default_graph()


app = Flask(__name__)


MODELS = ['30sec', '2min', '90min']


def clear_model_dir():
    dir_path = './model'
    for the_file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def copy_model_to_dir(train_time):
    src = f'./{train_time}Model'
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, './model')
        else:
            print('not file? ', full_file_name)


@app.route('/')
def index():
    train_time = request.args.get('trainTime') if request.args.get('trainTime') in MODELS else MODELS[0]
    clear_model_dir()
    copy_model_to_dir(train_time)
    setup_session()
    return render_template('index.html')


def bestmove(input):
    global graph
    with graph.as_default():
        data = (sess.run(tf.argmax(prediction.eval(session = sess, feed_dict={x:[input]}),1)))
    return data


@app.route('/api/ticky', methods=['POST'])
def ticky_api():
    data = request.get_json()
    data = np.array(data['data'])
    data = data.tolist()
    return jsonify(np.asscalar(bestmove(data)[0]))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

