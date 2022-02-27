from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy import misc

to_np = lambda x: x.data.cpu().numpy()
concat = lambda x: np.concatenate(x, axis=0)

def get_ood_scores_odin(loader, net, bs, ood_num_examples, T, noise, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    net.eval()
    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= ood_num_examples // bs and in_dist is False:
            break
        data = data.cuda()
        data = Variable(data, requires_grad = True)

        output = net(data)
        smax = to_np(F.softmax(output, dim=1))

        odin_score = ODIN(data, output,net, T, noise)
        _score.append(-np.max(odin_score, 1))

        if in_dist:
            preds = np.argmax(smax, axis=1)
            targets = target.numpy().squeeze()
            right_indices = preds == targets
            wrong_indices = np.invert(right_indices)

            _right_score.append(-np.max(smax[right_indices], axis=1))
            _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


def ODIN(inputs, outputs, model, temper, noiseMagnitude1):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    criterion = nn.CrossEntropyLoss()

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    
    gradient[:,0] = (gradient[:,0] )/(63.0/255.0)
    gradient[:,1] = (gradient[:,1] )/(62.1/255.0)
    gradient[:,2] = (gradient[:,2] )/(66.7/255.0)
    #gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
    #gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
    #gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

    return nnOutputs


def get_Mahalanobis_score(model, test_loader, num_classes, sample_mean, precision, layer_index, magnitude, num_batches, in_dist=False):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []
    Gassion_Entropy = []

    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx >= num_batches and in_dist is False:
            break


        
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        
        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean

            # print(torch.mm( torch.mm(zero_f, precision[layer_index]),  zero_f.t() ).shape)
            # batchsize * batchsize
            # exit(0)

            # torch.mm(a, b)是矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵
            # 如果输入是一个向量(1D 张量)，则返回一个以input为对角线元素的2D方阵
            # 如果输入是一个矩阵(2D 张量)，则返回一个包含input对角线元素的1D张量
            # 矩阵返回对角线元素, 对角线元素返回矩阵
            term_gau = -0.5*torch.mm( torch.mm(zero_f, precision[layer_index]),  zero_f.t() ).diag()
            # print(term_gau.shape)
            # batchsize * 1
            # exit(0)
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
            # print(term_gau)
            # print(term_gau.shape)
            # batchsize
            # exit(0)
        
        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
        # torch.ge(a, 0): 逐个元素和0比较大小
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
        

        tempInputs = torch.add(data.data, -magnitude, gradient)
        with torch.no_grad():
            noise_out_features = model.intermediate_forward(tempInputs, layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

        # print(noise_gaussian_score)
        # print(noise_gaussian_score.shape)
        # batch_size * num_sample_per_class
        # exit(0)
        gaussian_entropy_score = noise_gaussian_score.mean(1) - torch.logsumexp(noise_gaussian_score, dim=1) 
        # print(t.shape)
        # exit(0)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1) 
        # print(noise_gaussian_score.shape)
        # batch_size
        # exit(0)
        Mahalanobis.extend(-noise_gaussian_score.cpu().numpy())
        Gassion_Entropy.extend(-gaussian_entropy_score.cpu().numpy())
        
    return np.asarray(Mahalanobis, dtype=np.float32), np.asarray(Gassion_Entropy, dtype=np.float32)


def sample_estimator(model, num_classes, feature_list, train_loader):
    # feature_list: 特征的维数 = 128
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance
    
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    # print('num_output: ', num_output) # 1
    # exit(0)
    num_sample_per_class = np.empty(num_classes)
    # print(num_sample_per_class.shape) # (10,)
    # exit(0)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)
    # print(list_features)
    # [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    # exit(0)

    with torch.no_grad():
        for data, target in train_loader:
            total += data.size(0)
            data = data.cuda()
            # data = Variable(data, volatile=True)
            output, out_features = model.feature_list(data)
            # print('before :',out_features)
            # print(out_features[0].shape)
            # 2 * 128 * 8 * 8: batchsize * channel * length * height
            # print(len(out_features[0])) 
            # batch_size
            
            # get hidden features
            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                # 把feature拉直
                # print('type: ', type(out_features[i]))
                # print(out_features[i].shape)
                # 2 * 128  * 64
                # print('out_features[i].size(0): ', out_features[i].size(0))
                # batch_size: 2
                # print('out_features[i].size(1): ', out_features[i].size(1))
                # channel: 128
                # print("++++++++++++++++++++++++++++++++++")
                # print('after: ', out_features[i].shape)
                # 2 * 128 * 64
                # exit(0)
                # print(out_features[i])
                out_features[i] = torch.mean(out_features[i].data, 2)
                # 2 表示求第三维度的均值, 每个样本的特征的均值
                

                # print(out_features[i])
                # print(out_features[i].shape)
                # 2 * 128 每个通道的均值
                # exit(0)
                
            # compute the accuracy
            pred = output.data.max(1)[1]
            equal_flag = pred.eq(target.cuda()).cpu()
            correct += equal_flag.sum()

            # print(list_features)
            # [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            
            # construct the sample matrix
            for i in range(data.size(0)):
                label = target[i]
                if num_sample_per_class[label] == 0:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] = out[i].view(1, -1)
                        out_count += 1
                else:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] \
                        = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                        out_count += 1                
                num_sample_per_class[label] += 1
            # print(num_sample_per_class)
            # 每一类样本的数目
            # exit(0)
                
    sample_class_mean = []
    out_count = 0
    # t3 = 0
    for num_feature in feature_list:
        # t3 = t3 + 1
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1
    # print(sample_class_mean)
    # print(sample_class_mean[0].shape)
    # 10*128
    # exit(0)
    # print('t3: ', t3)
    # t3 = 1
    precision = []
    inv_precision = []

    for k in range(num_output):
        # 因为feature_list是列表(含有一个张量), 
        # 所以num_output = len(feature_list) = 1, num_out = 1, k = 0, 只会循环1次
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
        # print(X)
        # print(X.shape)
        # [5w,128]
        # exit(0)     
        # find inverse            
        group_lasso.fit(X.cpu().numpy())
        
        temp_precision = group_lasso.precision_
        

        temp_inv_precision = torch.from_numpy(np.linalg.inv(temp_precision)).float().cuda()
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        
        precision.append(temp_precision)
        # print(temp_precision.shape)
        # print(torch.det(temp_precision))
        # exit(0)
        inv_precision.append(temp_inv_precision)

        
    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision
    # return sample_class_mean, inv_precision
