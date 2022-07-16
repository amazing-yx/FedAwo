import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN, suma
from clients import ClientsGroup, client
import sys
import random


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
# parser.add_argument('-E', '--epoch', type=int, default=random.randrange(3, 6, 1), help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=150, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
panduan = 10


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__

    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger('noIID自适应+节省通讯fedavg0(统计客户端计算次数)1.txt')

    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    loss_func = F.cross_entropy
    net_sum = suma()
    net_sum = net_sum.to(dev)
    opti_sum = optim.SGD(net_sum.parameters(), lr=args['learning_rate'])
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

    myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    global_list = []
    global_loss_list = []
    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))

        # order = np.random.permutation(args['num_of_clients'])
        order = []
        for k in range(100):
            order.append(k)
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        client_num = 0
        sum_parameters = None
        for client in tqdm(clients_in_comm):
            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters)
            # 打印local_model 准确率情况
            torch.save(local_parameters, "./localmodel/model%d" % client_num)
            net.load_state_dict(local_parameters, strict=True)
            local_accu = 0
            local_num = 0
            # print('----------', local_accu, local_num)
            for data, label in testDataLoader:
                data, label = data.to(dev), label.to(dev)
                preds = net(data)
                torch.save(preds, "./tensor/%d,%d.pt" % (client_num, local_num))
                preds = torch.argmax(preds, dim=1)
                local_accu += (preds == label).float().mean()
                local_num += 1
            client_num += 1
            print('local_accuracy: {}'.format(local_accu / local_num))
            local_accuracy = float(float(local_accu) / float(local_num))
            # if sum_parameters is None:
            #     sum_parameters = {}
            #     for key, var in local_parameters.items():
            #         sum_parameters[key] = var.clone()
            # else:
            #     for var in sum_parameters:
            #         sum_parameters[var] = sum_parameters[var] + local_parameters[var]

        for epo in range(5):
            preds_i = 0
            preds_list = []
            for _, label in testDataLoader:
                label = label.to(dev)
                preds_list = []
                for preds_j in range(100):
                    preds_a = torch.load("./tensor/%d,%d.pt" % (preds_j, preds_i)).to(dev)
                    preds_list.append(preds_a)
                    preds_a = preds_a.t()
                preds_i += 1

                opti_sum.zero_grad()
                preds_list_tensor = torch.stack(preds_list, 0)
                preds_list_tensor = preds_list_tensor.permute(1, 2, 0)
                preds_z = net_sum(preds_list_tensor).to(dev)
                preds_z = preds_z.squeeze()
                '''print(preds_z.size())
                preds_z = preds_z.t()
                print(preds_z.size())'''
                loss = F.cross_entropy(preds_z, label)
                # print(loss)
                loss.backward()
                opti_sum.step()
        params = list(net_sum.named_parameters())                 # 输出权重矩阵
        w_list_tensor = params[0][1][0]
        print(w_list_tensor)
        # print('w_list_tensor_1:', type(w_list_tensor_1))
        # print('w_list_tensor_1[0]', w_list_tensor_1[0])


        for p in range(100):
            local_parameters = torch.load("./localmodel/model%d" % p)
            if p == 0:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone() * w_list_tensor[0]
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + (local_parameters[var] * w_list_tensor[p])

        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / sum(w_list_tensor))

        with torch.no_grad():
            if (i + 1) % args['val_freq'] == 0:
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    loss = F.cross_entropy(preds, label)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                print('global_accuracy: {}'.format(sum_accu / num))
                global_loss = float(loss)
                global_list.append(float(float(sum_accu) / float(num)))
                global_loss_list.append(global_loss)
        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args[
                                                                                                    'learning_rate'],
                                                                                                args[
                                                                                                    'num_of_clients'],
                                                                                                args['cfraction'])))
    print(global_list)
    print(global_loss_list)


