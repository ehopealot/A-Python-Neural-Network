from neuralNetwork import *
import unittest
import time

class TestHW2(unittest.TestCase):
    def setUp(self):
        set_threshold_value(1.0)
        data = []
        with open("hw2.csv", "r") as f:
            for l in f:
                data.append([x.replace('\r','').replace('\n','') for x in l.split(",")])
        self.nn = Neural_Network(1, data, eta = .3)


    def test_Uniform_Weights(self):
        self.nn.initialize_with_uniform_weight(.1)
        for row in self.nn.hidden_layer_weights:
            for col in row:
                self.assertEquals(.1, col)
        for row in self.nn.output_layer_weights:
            self.assertEquals(.1, col)
        self.assertEquals(3, len(self.nn.input_range))
        self.assertEquals(2, len(self.nn.hidden_node_range))

    def test_Random_Weights(self):
        self.nn.initialize_with_random_weights(-5, 5)
        for row in self.nn.hidden_layer_weights:
            for col in row:
                self.assertTrue(-5 <= col <= 5)
        for row in self.nn.output_layer_weights:
                self.assertTrue(-5 <= col <= 5)

    def test_Forward_propagate(self):
        self.nn.initialize_with_uniform_weight(.1)
        self.assertEqual(int(self.nn._forward_propagate(self.nn.data[0])*10**9), int(0.538668479964*10**9))

    def test_Backward_Propagate(self):
        self.nn.initialize_with_uniform_weight(.1)
        output = self.nn._forward_propagate(self.nn.data[0])
        self.nn._back_propagate(output, self.nn.data[0])
        error = self.nn.error
        hidden_node_error = self.nn.hidden_layer_errors[0]
        self.assertEqual(int(error * 10**9), int(0.114643073434*10**9))
        self.assertEqual(int(hidden_node_error * 10**9), int(0.0028376060621625464*10**9))

    def test_Update_Weights(self):
        self.nn.initialize_with_uniform_weight(.1)
        output = self.nn._forward_propagate(self.nn.data[0])
        self.nn._back_propagate(output, self.nn.data[0])
        self.nn._update_weights(output, self.nn.data[0])
        self.assertEqual(str(self.nn.hidden_layer_weights), "[[0.10085128181864877, 0.1, 0.10085128181864877]]")
        self.assertEqual(str(self.nn.output_layer_weights), "[0.1189103977991797, 0.1343929220303063]")



class TestIris(unittest.TestCase):
    def setUp(self):
        set_threshold_value(-1.0)
        data = []
        with open("iris.csv", "r") as f:
            for l in f:
                data.append([x.replace('\r','').replace('\n','') for x in l.split(",")])
        self.nn = Neural_Network(10, data, error_margin = .05)
#        nn.init_with_random_weights(-5, 5)
        self.nn.initialize_with_uniform_weight(.02)


    def test_iris_200(self):
        with open("output.txt", "w") as f:
            f.write('blah blah\n')
            nn = self.nn
            for i in range(200):
                self.nn.run_epoch()
#                f.write(str(nn))
                f.write("*** epoch " + str(i) + "***\n")
                f.write("*** max RMS: " + str(nn.epoch_results[i][0]) + " ***\n")
                f.write("*** avg RMS: " + str(nn.epoch_results[i][1]) + " ***\n")
                f.write("*** accuracy: " + str(nn.epoch_results[i][2]) + "% ***\n")
#*** max RMS: 0.322498441984 ***
#*** avg RMS: 0.0825405093718 ***
#*** accuracy: 0.453333333333% ***
            self.assertEqual(int(0.322498441984 * 10**10), int(nn.epoch_results[i][0]*10**10))
            self.assertEqual(int(0.0825405093718 * 10**10), int(nn.epoch_results[i][1]*10**10))
            self.assertEqual(int(0.453333333333*10**10), int(nn.epoch_results[i][2]*10**10))

class TestIris10000(unittest.TestCase):
    def setUp(self):
        
        data = []
        with open("iris.csv", "r") as f:
            for l in f:
                data.append([x.replace('\r','').replace('\n','') for x in l.split(",")])
        self.nn = Neural_Network(10, data, error_margin = .05)

    def test_iris_10000(self):
        x = time.time()
        self.nn.initialize_with_random_weights(-5, 5)
        for i in range(10000):
            if i%500 == 0: print i
            self.nn.run_epoch()
        print self.nn.epoch_results[9999]
        print time.time() - x
        





if __name__ == "__main__":
    unittest.main()
