#!/usr/bin/env python
"""Erik Hope --- CSC 578 --- Neural Network Midterm Project"""
from optparse import OptionParser
import random
import math
import sys
THRESHOLD_VALUE = -1.0

def set_threshold_value(value):
    """A function which sets the value
    of the threshold nodes in the neural network"""
    global THRESHOLD_VALUE
    THRESHOLD_VALUE = value


class Instance(object):
    """Contains some convenience functions for 
    dealing with instances. Takes a list of values, 
    assuming the last value is the target attribute, and 
    assigns the first n-1 elements plus a threshold value to the 
    attributes property, and the last value in the list to thetarget property"""
    def __init__(self, data):
         self._attributes = [float(x) for x in data[:-1]] + [THRESHOLD_VALUE]
         self._target = float(data[-1])
    @property
    def attributes(self):
        return self._attributes
    @property
    def target(self):
        return self._target


class Neural_Network():
    def make_input_node_matrix(self, data, hidden_nodes):
        """return a matrix where the number of rows is the number of hidden nodes
           and the number of columns is the number of input nodes
        """
        inputs = len(self.data[0].attributes)
        matrix = []
        for i in range(hidden_nodes):
            matrix.append([0]*inputs)
        return matrix
    def make_output_node_matrix(self, hidden_nodes):
        return [0] * (hidden_nodes + 1)


    def __init__(self, hidden_nodes, data, eta=.1, error_margin = .01,
                 momentum=0):
        self.data = []
        for d in data:
            self.data.append(Instance(d))
        self.eta = eta
        self.momentum = momentum
        self.error_margin = error_margin
        self.epochs = 0
        self.epoch_results = {}

        self.hidden_node_range = range(0, hidden_nodes+1)
        number_of_inputs = len(self.data[0].attributes)
        self.input_range = range(0, number_of_inputs)
        self.instance = 0
        
        self.hidden_layer_weights = self.make_input_node_matrix(self.data, hidden_nodes)
        self.output_layer_weights = self.make_output_node_matrix(hidden_nodes)

        self.hidden_layer_errors = [0] * hidden_nodes

        self.hidden_nodes = [0] * hidden_nodes + [THRESHOLD_VALUE]


    def _back_propagate(self, output, instance):
        """Perform the back propagation step"""
        self.error = output*(1.0-output)*(instance.target-output)
        hidden_layer_errors = self.hidden_layer_errors
        output_layer_weights = self.output_layer_weights
        hidden_nodes = self.hidden_nodes
        error = self.error
        for node in self.hidden_node_range[:-1]:
            node_value = hidden_nodes[node]
            hidden_layer_errors[node] = node_value * (1.0 - node_value) * output_layer_weights[node] * error
        
    def _forward_propagate(self, instance):
        """Forward propagates an instance through the network"""
        attributes = instance.attributes
        hidden_layer_weights = self.hidden_layer_weights
        hidden_nodes = self.hidden_nodes
        sigmoid = self._sigmoid
        input_range = self.input_range
        for row in self.hidden_node_range[:-1]:
            current_row = hidden_layer_weights[row]
            value = 0.0
            for cell in input_range:
                value += attributes[cell] * current_row[cell]
            self.hidden_nodes[row] = sigmoid(value)
        output_value = 0
        output_layer_weights = self.output_layer_weights
        for h in self.hidden_node_range:
            output_value += hidden_nodes[h] * output_layer_weights[h]
        return sigmoid(output_value)

    def _update_weights(self, output, instance):
        """Update weights based on errors computed in the back
        propagation step"""
        eta = self.eta
        error = self.error
        output_layer_weights = self.output_layer_weights
        hidden_nodes = self.hidden_nodes
        input_nodes = instance.attributes
        hidden_layer_weights = self.hidden_layer_weights
        hidden_layer_errors = self.hidden_layer_errors
        input_range = self.input_range
        for row in self.hidden_node_range:
            hidden_node_value = hidden_nodes[row]
            output_layer_weights[row] = eta*error*hidden_node_value + output_layer_weights[row]
        for row in self.hidden_node_range[:-1]:
            hidden_node = hidden_layer_weights[row]
            error = hidden_layer_errors[row]
            for cell in input_range:
                input_node_value = input_nodes[cell]
                hidden_node[cell] = eta*error*input_node_value + hidden_node[cell]

    def _sigmoid(self, value):
        return 1.0/(1.0 + math.e**(-value))

    def initialize_with_random_weights(self, minimum=0, maximum=.2):
        for row in self.hidden_layer_weights:
            for cell in self.input_range:
                row[cell] = random.uniform(minimum, maximum)
        for cell in self.hidden_node_range:
            self.output_layer_weights[cell] = random.uniform(minimum, maximum)

    def initialize_with_uniform_weight(self, weight=.1):
        for row in self.hidden_layer_weights:
            for cell in self.input_range:
                row[cell] = weight
        for cell in self.hidden_node_range:
            self.output_layer_weights[cell] = weight

    def evaluate_instance(self, input_instance):
        """Once the network is trained, this method can be used to
        evaluate new instances"""
        instance = Instance(input_instance)
        return self._forward_propagate(instance)

    def run_epoch(self):
        """Performs all of the tasks 
        required in running an epoch during training:
        forward propagation, back propagation and
        weight update. stores the result in epoch_results"""
        correct = 0
        RMSes = []
        for instance in self.data:
            output = self._forward_propagate(instance)
            self._back_propagate(output, instance)
            self._update_weights(output, instance)
        for instance in self.data:
            #### Determine the accuracy ####
            output = self._forward_propagate(instance)
            RMS = math.sqrt((instance.target - output)**2) 
            RMSes.append(RMS)
            if RMS < self.error_margin:
                correct += 1

        self.epoch_results[self.epochs] = (max(RMSes), sum(RMSes)/float(len(RMSes)), 
                                           float(correct)/len(self.data))
        self.epochs += 1


if __name__ == "__main__":
    usage = "usage: %prog [options] INPUT_FILE"
    parser = OptionParser(usage)
    parser.add_option("-n", "--hidden_nodes", dest="hidden_nodes",
                      help="the number of hidden nodes (default 10)", default=10, type="int")
    parser.add_option("-l", "--learning_rate", dest="learning_rate",
                      help="the learning rate for the network (default .1)", default=.1, type="float")
#### Momentum Not Implemented Yet ####
#    parser.add_option("-m", "--momentum", dest="momentum",
#                      help="the momentum of the network", default=0, type="float")
    parser.add_option("-e", "--epochs", dest="epochs",
                      help="the number of epochs before terminating (default 10000)", default=10000, type="int")
    parser.add_option("-a", "--accuracy", dest="accuracy",
                      help="the accuracy needed before terminating (default 100)", default=100, type="float")
    parser.add_option("-d", "--delimiter", dest="delimiter",
                      help="delimiter of attributes in input file (default ',')", default=",")
    parser.add_option("-r", "--error_margin", dest="error_margin",
                      help="the error margin for training (default .05)", default=.05, type="float")
    parser.add_option("-w", "--start_weights", dest="start_weights",
                       help="""the initial weights of the network, one value if all weights are to be same,
                               comma delimited [n,n] if the weights should be initialized to a random value
                               in a range (default -5,5)""", default="-5,5")
    parser.add_option("-o", "--output_file", dest="output_file", 
                      help="""The file to which output is written (default stdout)""", default=None)
    (options, args) = parser.parse_args()
    
    data = []
    if len(args) == 0: 
        print "provide an input file!"
        sys.exit(0)
    with open(args[0], "r") as f:
        for l in f:
            data.append([x.replace('\r','').replace('\n','') for x in l.split(",")])

    nn = Neural_Network(options.hidden_nodes, data, options.learning_rate, 
                        options.error_margin)
    weights = [float(x) for x in options.start_weights.split(",")]
    if len(weights) == 1:
        nn.initialize_with_uniform_weight(*weights)
    elif len(weights) == 2:
        nn.initialize_with_random_weights(*weights)
    else:
        print "provide proper input weights!"
        sys.exit(0)
    accuracy = options.accuracy

    old_stdout = None
    outp_file = None
    if options.output_file:
        outp_file = open(options.output_file, "w")
        old_stdout = sys.stdout
        sys.stdout = outp_file

    for i in range(options.epochs):
        nn.run_epoch()
        print "***** Epoch %s *****" % (i + 1)
        print "Maximum RMSE: %s" % nn.epoch_results[i][0]
        print "Average RMSE: %s" % nn.epoch_results[i][1]
        print "Percent Correct: %s" % nn.epoch_results[i][2]
        if nn.epoch_results[i][2] *100 >= accuracy:
            break

    if outp_file:
        outp_file.close()
        sys.stdout = old_stdout
    print """Neural Network Trained. Enter a comma delimited instance or type \"quit\" to quit."""

    inp = ""
    quit = False
    while not quit:
        inp = raw_input("... ")
        try:
            print nn.evaluate_instance([x.replace('\n','').replace('\r','') for x in inp.split(",")] + [1])
        except Exception as e:
            quit = ('quit' == inp.lower().strip())
            if not quit:
                print "Input a valid instance!"
                
        



    
