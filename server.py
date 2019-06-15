# importing Flask
from flask import Flask, jsonify, render_template, request

#import json
# import tic tac toe game 
from deep_reinforcement_learning import *

import numpy as np


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


@app.route('/')
def index():
    return render_template('index.html')


def  bestmove(input):
    global graph
    with graph.as_default():
        data = (sess.run(tf.argmax(prediction.eval(session = sess,feed_dict={x:[input]}),1)))
    return data


@app.route('/api/ticky', methods=['POST'])
def ticky_api():
    data = request.get_json()
    data = np.array(data['data'])
    data = data.tolist()
    #print('data is ')
    #print(type(data))
    #print(data)
    return jsonify(np.asscalar(bestmove(data)[0]))


if __name__ == '__main__':
    #app.run(host='0.0.0.0',port=80,debug=True)
    #app.run(host='10.7.129.166',port=5000)
    app.run(host='0.0.0.0', port=5000)

