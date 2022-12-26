#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch, pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy, math, pdb, sys
import time, importlib
from DatasetLoader import test_dataset_loader
from torch.cuda.amp import autocast, GradScaler
import cv2
import glob
from PIL import Image

from flask import Flask, request
import requests

class EmbedNet(nn.Module):

    def __init__(self, model, optimizer, trainfunc, nPerClass, **kwargs):
        super(EmbedNet, self).__init__();

        ## __S__ is the embedding model
        EmbedNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__S__ = EmbedNetModel(**kwargs);

        ## __L__ is the classifier plus the loss function
        LossFunction = importlib.import_module('loss.'+trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs);

        ## Number of examples per identity per batch
        self.nPerClass = nPerClass

    def forward(self, data, label=None):

        data    = data.reshape(-1,data.size()[-3],data.size()[-2],data.size()[-1])
        outp    = self.__S__.forward(data)

        if label == None:
            return outp

        else:
            outp    = outp.reshape(self.nPerClass,-1,outp.size()[-1]).transpose(1,0).squeeze(1)
            nloss, prec1 = self.__L__.forward(outp,label)
            return nloss, prec1


class ModelTrainer(object):

    def __init__(self, embed_model, optimizer, scheduler, mixedprec, **kwargs):

        self.__model__  = embed_model

        ## Optimizer (e.g. Adam or SGD)
        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)

        ## Learning rate scheduler
        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        ## For mixed precision training
        self.scaler = GradScaler() 
        self.mixedprec = mixedprec

        assert self.lr_step in ['epoch', 'iteration']

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, verbose):

        self.__model__.train();

        stepsize = loader.batch_size;

        counter = 0;
        index   = 0;
        loss    = 0;
        top1    = 0     # EER or accuracy

        tstart = time.time()
        
        for data, label in loader:

            data    = data.transpose(1,0)

            ## Reset gradients
            self.__model__.zero_grad();

            ## Forward and backward passes
            if self.mixedprec:
                with autocast():
                    nloss, prec1 = self.__model__(data.to('cuda:1'), label.to('cuda:1'))
                self.scaler.scale(nloss).backward();
                self.scaler.step(self.__optimizer__);
                self.scaler.update();       
            else:
                nloss, prec1 = self.__model__(data.to('cuda:1'), label.to('cuda:1'))
                nloss.backward();
                self.__optimizer__.step();

            loss    += nloss.detach().cpu();
            top1    += prec1.detach().cpu();
            counter += 1;
            index   += stepsize;

            telapsed = time.time() - tstart
            tstart = time.time()

            if verbose:
                sys.stdout.write("\rProcessing (%d) "%(index));
                sys.stdout.write("Loss %f TEER/TAcc %2.3f%% - %.2f Hz "%(loss/counter, top1/counter, stepsize/telapsed));
                sys.stdout.flush();

            if self.lr_step == 'iteration': self.__scheduler__.step()

        if self.lr_step == 'epoch': self.__scheduler__.step()

        sys.stdout.write("\n");
        
        return (loss/counter, top1/counter);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, transform, print_interval=100, num_eval=10, **kwargs):
        
        self.__model__.eval();
        
        feats       = {}
        tstart      = time.time()

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = sum([x.strip().split(',')[-2:] for x in lines],[])
        setfiles = list(set(files))
        setfiles.sort()

        ## Define test data loader
        test_dataset = test_dataset_loader(setfiles, test_path, transform=transform, num_eval=num_eval, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=nDataLoaderThread,
            drop_last=False,
        )

        ## Extract features for every image
        for idx, data in enumerate(test_loader):
            inp1                = data[0][0].to('cuda:1')
            ref_feat            = self.__model__(inp1).detach().cpu()
            feats[data[1][0]]   = ref_feat
            telapsed            = time.time() - tstart

            if idx % print_interval == 0:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(setfiles),idx/telapsed,ref_feat.size()[1]));

        print('')
        all_scores = [];
        all_labels = [];
        tstart = time.time()

        ## Read files and compute all scores
        for idx, line in enumerate(lines):

            data = line.strip().split(',');

            ref_feat = feats[data[1]]
            com_feat = feats[data[2]]

            score = F.cosine_similarity(ref_feat, com_feat)

            all_scores.append(score);  
            all_labels.append(int(data[0]));

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx,len(lines),idx/telapsed));
                sys.stdout.flush();

        print('')

        return (all_scores, all_labels);

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Deploy server
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def deploy_server(self, port, transform=None):
        
        self.__model__.eval();

        sys.path.append('detectors')
        from detectors import S3FD

        DET = S3FD(device='cuda')

        # create the Flask app
        app = Flask(__name__)

        @app.route('/query', methods=['POST'])
        def query():

            # unpack the received data
            data = pickle.loads(request.get_data())

            # image save
            data['img'].save("tmp1.jpg")

            
            iname = data['name']

            image = cv2.imread("tmp1.jpg")

            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[0.5])

            if len(bboxes) != 1:
                return "fail"

            bsi = 100

            sx = int((bboxes[0][0]+bboxes[0][2])/2) + bsi
            sy = int((bboxes[0][1]+bboxes[0][3])/2) + bsi
            ss = int(max((bboxes[0][3]-bboxes[0][1]),(bboxes[0][2]-bboxes[0][0]))/2)

            image = numpy.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))

            face = image[int(sy-ss):int(sy+ss),int(sx-ss):int(sx+ss)]
            face = cv2.resize(face,(240,240))

            cv2.imwrite('tmp2.jpg'.format(iname),face)

            im1 = Image.open('tmp2.jpg'.format(iname)) 

            inp1 = transform(im1).to('cuda:1')

            com_feat            = self.__model__(inp1).detach().cpu()

            files = glob.glob('saved_feats/*.pt')

            max_score = 0
            for file in files:

                ref_feat = torch.load(file)

                score = F.cosine_similarity(ref_feat, com_feat)

                if(score>max_score) :
                    max_score = score
                    max_file = file

                print('{} {:.2f}'.format(file,score.item()))

            # sx ex sy ey
            return str(bboxes[0][0]) + " " + str(bboxes[0][2]) + " " + str(bboxes[0][1]) + " " + str(bboxes[0][3]) + " "+ str('{} {:.2f}'.format(max_file, max_score.item()))

        @app.route('/enroll', methods=['POST'])
        def enroll():

            # unpack the received data
            data = pickle.loads(request.get_data())

            data['img'].save("tmp1.jpg")

            iname = data['name']

            image = cv2.imread("tmp1.jpg")

            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[0.5])

            bsi = 100

            sx = int((bboxes[0][0]+bboxes[0][2])/2) + bsi
            sy = int((bboxes[0][1]+bboxes[0][3])/2) + bsi
            ss = int(max((bboxes[0][3]-bboxes[0][1]),(bboxes[0][2]-bboxes[0][0]))/2)

            image = numpy.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))

            face = image[int(sy-ss):int(sy+ss),int(sx-ss):int(sx+ss)]
            face = cv2.resize(face,(240,240))

            cv2.imwrite('saved_feats/{}.jpg'.format(iname),face)

            im1 = Image.open('saved_feats/{}.jpg'.format(iname)) 

            inp1 = transform(im1).to('cuda:1')

            ref_feat            = self.__model__(inp1).detach().cpu()

            torch.save(ref_feat,'saved_feats/{}.pt'.format(iname))

            return "success"


        app.run(host='0.0.0.0', debug=True, port=port, threaded=False)



    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):
        
        torch.save(self.__model__.state_dict(), path);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.__model__.state_dict();
        loaded_state = torch.load(path);
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                if name not in self_state:
                    print("%s is not in the model."%origname);
                    continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);

