from __future__ import print_function, division
import pickle
from pdb import set_trace
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import csv
from collections import Counter
from sklearn import svm
import matplotlib.pyplot as plt
import time
import os
import pandas as pd


class MAR(object):
    def __init__(self):
        self.fea_num = 4000
        self.step = 100
        self.enough = 20
        self.kept=50
        self.atleast=100
        self.syn_thres = 0.8
        self.enable_est = True
        self.interval = 5
        self.crash = 'no'
        self.norm = 'l2row'
        self.metrics = 'no'
        self.false_neg = 0.5
        self.false_alarm = 0
        self.correction = 'no' # 'machine' or 'knee'
        self.neg_len = 0.5



    def create(self,filename, type = 'all'):
        self.filename=filename
        self.name=self.filename.split(".")[0]
        self.flag=True
        self.hasLabel=True
        self.record={"x":[],"pos":[]}
        self.body={}
        self.est=[]
        self.last_pos=0
        self.last_neg=0
        self.record_est={"x":[],"semi":[]}
        self.round = 0
        self.target_vul_type = type

        try:
            ## if model already exists, load it ##
            return self.load()
        except:
            # print("Loading data")
            ## otherwise read from file ##

            self.loadfile()
            # print("Preprocessing")
            self.preprocess()
                # self.lda()
                # self.save()

        return self

    ### Use previous knowledge, labeled only
    def create_old(self, filename):
        with open("../workspace/coded/" + str(filename), "r") as csvfile:
            content = [x for x in csv.reader(csvfile, delimiter=',')]
        fields = ["Document Title", "Abstract", "Year", "PDF Link", "code", "time"]
        header = content[0]
        ind0 = header.index("code")
        self.last_pos = len([c[ind0] for c in content[1:] if c[ind0] == "yes"])
        self.last_neg = len([c[ind0] for c in content[1:] if c[ind0] == "no"])
        for field in fields:
            ind = header.index(field)
            if field == "time":
                self.body[field].extend([float(c[ind]) for c in content[1:] if c[ind0] != "undetermined"])
            else:
                self.body[field].extend([c[ind] for c in content[1:] if c[ind0] != "undetermined"])
        try:
            ind = header.index("label")
            self.body["label"].extend([c[ind] for c in content[1:] if c[ind0]!="undetermined"])
        except:
            self.body["label"].extend([c[ind0] for c in content[1:] if c[ind0]!="undetermined"])

        self.preprocess()
        # self.save()

    ### Use previous knowledge, pos only
    def create_pos(self, filename):
        with open("../workspace/coded/" + str(filename), "r") as csvfile:
            content = [x for x in csv.reader(csvfile, delimiter=',')]
        fields = ["Document Title", "Abstract", "Year", "PDF Link", "code", "time"]
        header = content[0]
        ind0 = header.index("code")
        self.last_pos = len([c[ind0] for c in content[1:] if c[ind0] == "yes"])
        self.last_neg = 0
        for field in fields:
            ind = header.index(field)
            if field == "time":
                self.body[field].extend([float(c[ind]) for c in content[1:] if c[ind0] == "yes"])
            else:
                self.body[field].extend([c[ind] for c in content[1:] if c[ind0] == "yes"])
        try:
            ind = header.index("label")
            self.body["label"].extend([c[ind] for c in content[1:] if c[ind0]=="yes"])
        except:
            self.body["label"].extend([c[ind0] for c in content[1:] if c[ind0]=="yes"])

        self.preprocess()
        # self.save()


    def loadfile(self):
        try:
            self.body = pd.read_csv('/share/tjmenzie/zyu9/data/' + self.filename)
        except:
            self.body = pd.read_csv('../../Datasets/vulns/' + self.filename)


        if self.target_vul_type == 'all':
            label = ['yes' if int(x)>0 else 'no' for x in self.body['severity']]
        else:
            label = ['yes' if not pd.isnull(types) and self.target_vul_type in set(types.split(',')) else 'no' for types in self.body['type']]
        self.body['label']=pd.Series(label, index=self.body.index)
        self.body['code']=pd.Series(['undetermined']*len(label), index=self.body.index)
        self.body['time']=pd.Series([0]*len(label), index=self.body.index)
        self.body['fixed']=pd.Series([0]*len(label), index=self.body.index)
        self.body['count']=pd.Series([0]*len(label), index=self.body.index)

        return
    
    def lda(self):
        import lda

        tfer = TfidfVectorizer(lowercase=False, stop_words="english", norm=None, use_idf=False,
                               decode_error="ignore")
        self.csr_mat = tfer.fit_transform(self.body['sourcecode'])

        if self.crash == 'append':
            from scipy import sparse
            self.csr_mat = sparse.hstack((self.csr_mat,np.array(self.body['crashes'])[:,None]))

        lda1 = lda.LDA(n_topics=100, alpha=0.1, eta=0.01, n_iter=200)
        self.csr_mat = lda1.fit_transform(self.csr_mat.astype(int))
        self.csr_mat = preprocessing.normalize(self.csr_mat,norm='l2',axis=1)
        return

    def plot_lda(self):
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 20}

        plt.rc('font', **font)
        paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
                 'figure.autolayout': True, 'figure.figsize': (16, 8)}

        plt.rcParams.update(paras)

        fig = plt.figure()

        poses = np.where(np.array(self.body['label']) == "yes")[0]
        negs = np.where(np.array(self.body['label']) == "no")[0]
        pos_lda = np.sum(self.csr_mat[poses],axis=0)/len(poses)
        neg_lda = np.sum(self.csr_mat[negs],axis=0)/len(negs)
        m=self.csr_mat.shape[1]


        plt.plot(range(m), pos_lda)
        plt.plot(range(m), neg_lda)
        name=self.name+ "_" + str(int(time.time()))+".png"

        dir = "./static/image"
        for file in os.listdir(dir):
            os.remove(os.path.join(dir, file))

        plt.savefig("./static/image/" + name)
        plt.close(fig)
        return name


    def doc2vec(self):
        from gensim.models import Doc2Vec
        from gensim.models.doc2vec import TaggedDocument
        import multiprocessing

        def convert_sentences(sentence_list):
            for i in range(len(sentence_list)):
                for char in ['.', ',', '!', '?', ';', ':']:
                    sentence_list[i] = sentence_list[i].replace(char, ' ' + char + ' ')
            return [TaggedDocument(words=sentence_list[i].split(), tags=[i]) for i in range(len(sentence_list))]

        def normalize(x, p=2):
            xx = np.linalg.norm(x, p)
            return x / xx if xx else x

        content = [self.body["Document Title"][index] + " " + self.body["Abstract"][index] for index in
                   xrange(len(self.body["Document Title"]))]

        content1 = convert_sentences(content)
        model = Doc2Vec(size=300, window=10, min_count=5, workers=multiprocessing.cpu_count(),alpha=0.025, min_alpha=0.025)
        model.build_vocab(content1)

        for epoch in range(10):
            model.train(content1, total_examples=model.corpus_count, epochs=model.iter)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay

        self.csr_mat = np.array([normalize(model.infer_vector(x.words, alpha=model.alpha, min_alpha=model.min_alpha),p=2) for x in content1])
        return
    
    def syn_error(self):
        tmp = [Counter([self.body['label'][j] for j, vecb in enumerate(self.csr_mat) if
                                           (vec * vecb.transpose()).toarray()[0, 0] >= self.syn_thres]) for vec in
                                  self.csr_mat]
        self.body['syn_error'] = [x['yes']/sum(x.values()) for x in tmp]
        fields = ["Document Title", "Abstract", "Year", "PDF Link", "label", "syn_error"]
        with open("../workspace/data/" + str(self.name) + "_error.csv", "wb") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(fields)
            for ind in xrange(len(self.body['label'])):
                csvwriter.writerow([self.body[field][ind] for field in fields])
        return

    def export_feature(self):
        with open("../workspace/coded/feature_" + str(self.name) + ".csv", "wb") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            for i in xrange(self.csr_mat.shape[0]):
                for j in range(self.csr_mat.indptr[i],self.csr_mat.indptr[i+1]):
                    csvwriter.writerow([i+1,self.csr_mat.indices[j]+1,self.csr_mat.data[j]])
        return

    def get_numbers(self):
        total = len(self.body["code"]) - self.last_pos - self.last_neg
        pos = Counter(self.body["code"])["yes"] - self.last_pos
        neg = Counter(self.body["code"])["no"] - self.last_neg
        try:
            tmp=self.record['x'][-1]
        except:
            tmp=-1
        if int(pos+neg)>tmp:
            self.record['x'].append(int(pos+neg))
            self.record['pos'].append(int(pos))
        self.pool = np.where(np.array(self.body['code']) == "undetermined")[0]
        self.labeled = list(set(range(len(self.body['code']))) - set(self.pool))
        return pos, neg, total

    def export(self):
        fields = ["Document Title", "Abstract", "Year", "PDF Link", "label", "code","time"]
        with open("../workspace/coded/" + str(self.name) + ".csv", "wb") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(fields)
            ## sort before export
            time_order = np.argsort(self.body["time"])[::-1]
            yes = [c for c in time_order if self.body["code"][c]=="yes"]
            no = [c for c in time_order if self.body["code"][c] == "no"]
            und = [c for c in time_order if self.body["code"][c] == "undetermined"]
            ##
            for ind in yes+no+und:
                csvwriter.writerow([self.body[field][ind] for field in fields])
        return

    def preprocess(self):

        if self.metrics=='only':
            self.csr_mat = self.body[['CountClassBase', 'CountClassCoupled','CountClassDerived','CountDeclInstanceVariablePrivate','CountDeclMethod','CountInput','CountLine','CountOutput','Cyclomatic','MaxInheritanceTree','MaxNesting']].as_matrix()
            return

        #######################################################

        ### Feature selection by tfidf in order to keep vocabulary ###
        tfidfer = TfidfVectorizer(lowercase=False, stop_words=None, norm=None, use_idf=True, smooth_idf=False,
                                sublinear_tf=False,decode_error="ignore",max_features=4000)

        # tfidfer = TfidfVectorizer(lowercase=False, stop_words=None, norm=None, use_idf=True, smooth_idf=False,
        #                         sublinear_tf=False,decode_error="ignore",max_features=4000,token_pattern='(?u)\b(\w*\.*\w+)+\b',analyzer='word')

        self.body['sourcecode'].fillna('', inplace=True)

        tfidfer.fit(self.body['sourcecode'])

        self.voc = tfidfer.vocabulary_.keys()

        ##############################################################

        ### Term frequency as feature, L2 normalization ##########
        tfer = TfidfVectorizer(lowercase=True, stop_words=None, norm=None, use_idf=False,
                        vocabulary=self.voc,decode_error="ignore")
        # tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False,
        #                 vocabulary=self.voc,decode_error="ignore")
        self.csr_mat=tfer.fit_transform(self.body['sourcecode'])

        if self.norm=='l2rowcol':
            self.csr_mat = preprocessing.normalize(self.csr_mat,norm='l2',axis=1)

        if self.crash == 'append':
            from scipy import sparse
            self.csr_mat = sparse.hstack((self.csr_mat,np.array(self.body['crashes'])[:,None]))

        if self.norm=='l2row':
            self.csr_mat = preprocessing.normalize(self.csr_mat,norm='l2',axis=1)
        elif self.norm=='l2col' or self.norm=='l2rowcol':
            self.csr_mat = preprocessing.normalize(self.csr_mat,norm='l2',axis=0)
        elif self.norm=='pca':
            from sklearn.decomposition import PCA
            norm = PCA(n_components=100)
            self.csr_mat = norm.fit_transform(self.csr_mat.todense())
        elif self.norm=='l1col':
            self.csr_mat = preprocessing.normalize(self.csr_mat,norm='l1',axis=0)

        ########################################################
        return

    ## save model ##
    def save(self):
        with open("memory/"+str(self.name)+".pickle","w") as handle:
            pickle.dump(self,handle)

    ## load model ##
    def load(self):
        with open("memory/" + str(self.name) + ".pickle", "r") as handle:
            tmp = pickle.load(handle)
        return tmp

    def knee_error(self,clf):
        coded = np.where(np.array(self.body['code']) != "undetermined")[0]
        neg_at = list(clf.classes_).index("no")
        order = np.argsort(clf.predict_proba(self.csr_mat[coded])[:,neg_at])
        labels = self.body['code'][coded][order]
        poses = []
        count = 0
        for l in labels:
            if l=='yes':
                count+=1
            poses.append(count)
        xes = np.array(range(len(coded)))+1


        y = poses[-1]
        s = xes[-1]
        ratio = s/np.sqrt(y**2+s**2)
        per_best=-1
        best=0
        for i in range(len(xes))[::-1]:
            per = (poses[i]-xes[i]*y/s)*ratio

            if per>per_best:
                best = i
                per_best=per
        sel = [x for x in coded[order][:best+1] if self.body['code'][x]=='no' and self.body['fixed'][x]==0]
        return sel, np.array([1]*len(sel))



    def knee(self):
        y = self.record['pos'][-1]
        s = self.record['x'][-1]
        ratio = s/np.sqrt(y**2+s**2)
        per_best=-1
        best=0
        for i in range(len(self.record['x']))[::-1]:
            per = (self.record['pos'][i]-self.record['x'][i]*y/s)*ratio

            if per>per_best:
                best = i
                per_best=per
        self.kneepoint = best
        rho = (s-self.record['x'][best])*self.record['pos'][best]/self.record['x'][best]/(1+y-self.record['pos'][best])
        # thres = 156-min((150,y))
        thres = 6
        if rho>=thres:
            return True
        else:
            return False

    def estimate_curve(self, clf, reuse=False, num_neg=0):
        from sklearn import linear_model
        import random


        # def prob_sample(probs):
        #     order = np.argsort(probs)[::-1]
        #     count = 0
        #     can = []
        #     sample = []
        #     for i, x in enumerate(probs[order]):
        #         count = count + x
        #         can.append(order[i])
        #         if count >= 1:
        #             # sample.append(np.random.choice(can,1)[0])
        #             sample.append(can[0])
        #             count = 0
        #             can = []
        #     return sample

        def prob_sample(probs):
            order = np.argsort(probs)[::-1]
            count = 0
            can = []
            sample = []
            where = 1
            for i, x in enumerate(probs[order]):
                count = count + x
                can.append(order[i])
                if count >= where:
                    # sample.append(np.random.choice(can,1)[0])
                    sample.append(can[0])
                    where = where + 1
                    can = []
            return sample

        # def prob_sample(probs):
        #     sample=[]
        #     for i,x in enumerate(probs):
        #         if random.random<x:
        #             sample.append(i)
        #     return sample

        ### just labeled

        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]

        poses = np.array(poses)[np.argsort(np.array(self.body['time'])[poses])[self.last_pos:]]
        negs = np.array(negs)[np.argsort(np.array(self.body['time'])[negs])[self.last_neg:]]

        ###############################################

        # prob = clf.predict_proba(self.csr_mat)[:,:1]
        prob1 = clf.decision_function(self.csr_mat)
        prob = np.array([[x] for x in prob1])


        y = np.array([1 if x == 'yes' else 0 for x in self.body['code']])
        y0 = np.copy(y)

        if len(poses) and reuse:
            all = list(set(poses) | set(negs) | set(self.pool))
        else:
            all = range(len(y))
        ####
        # overweight=1
        # all=overweight*(list(poses)+list(negs))+all
        ####
        ####
        ####

        ##
        # C = Counter(y[all])[1] / num_neg
        # es = linear_model.LogisticRegression(penalty='l2', fit_intercept=True, C=C)
        #
        # es.fit(prob[all], y[all])
        # pos_at = list(es.classes_).index(1)
        #
        # pre0 = es.predict_proba(prob[self.pool])[:, pos_at]
        ##


        pos_num_last = Counter(y0)[1]

        lifes = 1
        life = lifes
        pos_num = Counter(y0)[1]

        while (True):
            C = Counter(y[all])[1]/ num_neg
            es = linear_model.LogisticRegression(penalty='l2', fit_intercept=True, C=C)

            es.fit(prob[all], y[all])
            pos_at = list(es.classes_).index(1)


            pre = es.predict_proba(prob[self.pool])[:, pos_at]

            # pre= pre0*0.1+pre*0.9
            # pre0=pre

            y = np.copy(y0)

            sample = prob_sample(pre)
            for x in self.pool[sample]:
                y[x] = 1

            # for x in self.pool[np.argsort(pre)[-int(sum(pre)):]]:
            #     y[x]=1

            pos_num = Counter(y)[1]
            # crit=(pos_num-Counter(y0)[1])/pos_num
            # crit = (pos_num - pos_num_last) / pos_num_last
            # if crit<0.05:
            #     break
            if pos_num == pos_num_last:
                life = life - 1
                if life == 0:
                    break
            else:
                life = lifes
            pos_num_last = pos_num
        esty = pos_num - self.last_pos
        pre = es.predict_proba(prob)[:, pos_at]

        ###
        # pre2 = es.predict_proba(prob[self.pool])[:, pos_at]
        # y = np.copy(y0)
        # for x in self.pool[np.argsort(pre2)[len(poses):]]:
        #     y[x] = 1
        # es.fit(prob, y)
        # pos_at = list(es.classes_).index(1)
        # pre2 = es.predict_proba(prob)[:, pos_at]
        ###

        ##### simu curve #######
        # self.simcurve={'x':[self.record['x'][-1]],'pos':[self.record['pos'][-1]]}
        # already=decayed
        # pool=np.where(np.array(self.body['code']) == "undetermined")[0]
        # clff=svm.SVC(kernel='linear', probability=True)
        # while True:
        #     clff.fit(self.csr_mat[already], y[already])
        #     pos_at = list(clff.classes_).index(1)
        #     prob = clff.predict_proba(self.csr_mat[pool])[:, pos_at]
        #     sample = pool[np.argsort(prob)[::-1][:self.step]]
        #     already = already+list(sample)
        #     pool = np.array(list(set(pool)-set(sample)))
        #     self.simcurve['x'].append(self.simcurve['x'][-1]+self.step)
        #     self.simcurve['pos'].append(Counter(y[already])[1])
        #     if self.simcurve['pos'][-1] > int(Counter(y)[1]*0.9) or self.simcurve['pos'][-1]==self.simcurve['pos'][-2]:
        #         break
        # set_trace()
        ########################


        ########## inspect curve
        # font = {'family': 'normal',
        #         'weight': 'bold',
        #         'size': 20}
        #
        # plt.rc('font', **font)
        # paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
        #          'figure.autolayout': True, 'figure.figsize': (16, 8)}
        #
        # plt.rcParams.update(paras)
        #
        # fig = plt.figure()
        # plt.scatter(prob1[self.pool], y[self.pool], marker='.', s=500, color='0.75')
        # plt.scatter(prob1[poses], y[poses], marker='o', s=500, color='blue')
        # plt.scatter(prob1[negs], y[negs], marker='x', s=500, color='red')
        # order = np.argsort(prob1[all])
        # plt.plot(prob1[all][order],pre[all][order], color='black')
        #
        # plt.ylabel("Prediction")
        # plt.xlabel("Labels")
        # name = self.name + "_" + str(int(time.time())) + ".png"
        #
        # dir = "./static/image"
        # for file in os.listdir(dir):
        #     os.remove(os.path.join(dir, file))
        #
        # plt.savefig("./static/image/" + name)
        # plt.close(fig)
        ###########
        return esty, pre

    ## BM25 ##
    def BM25(self,query):
        b=0.75
        k1=1.5

        ### Combine title and abstract for training ###########
        content = [self.body["Document Title"][index] + " " + self.body["Abstract"][index] for index in
                   xrange(len(self.body["Document Title"]))]
        #######################################################

        ### Feature selection by tfidf in order to keep vocabulary ###

        tfidfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False, smooth_idf=False,
                                  sublinear_tf=False, decode_error="ignore")
        tf = tfidfer.fit_transform(content)
        d_avg = np.mean(np.sum(tf, axis=1))
        score = {}
        for word in query:
            score[word]=[]
            try:
                id= tfidfer.vocabulary_[word]
            except:
                score[word]=[0]*len(content)
                continue
            df = sum([1 for wc in tf[:,id] if wc>0])
            idf = np.log((len(content)-df+0.5)/(df+0.5))
            for i in xrange(len(content)):
                score[word].append(idf*tf[i,id]/(tf[i,id]+k1*((1-b)+b*np.sum(tf[0],axis=1)[0,0]/d_avg)))
        self.bm = np.sum(score.values(),axis=0)

    def BM25_get(self):
        return self.pool[np.argsort(self.body['crashes'][self.pool])[::-1][:self.step]]

    def estimate_curve(self, clf, reuse=False, num_neg=0):
        from sklearn import linear_model
        import random


        # def prob_sample(probs):
        #     order = np.argsort(probs)[::-1]
        #     count = 0
        #     can = []
        #     sample = []
        #     for i, x in enumerate(probs[order]):
        #         count = count + x
        #         can.append(order[i])
        #         if count >= 1:
        #             # sample.append(np.random.choice(can,1)[0])
        #             sample.append(can[0])
        #             count = 0
        #             can = []
        #     return sample

        def prob_sample(probs):
            order = np.argsort(probs)[::-1]
            count = 0
            can = []
            sample = []
            where = 1
            for i, x in enumerate(probs[order]):
                count = count + x
                can.append(order[i])
                if count >= where:
                    # sample.append(np.random.choice(can,1)[0])
                    sample.append(can[0])
                    where = where + 1
                    can = []
            return sample

        # def prob_sample(probs):
        #     sample=[]
        #     for i,x in enumerate(probs):
        #         if random.random<x:
        #             sample.append(i)
        #     return sample

        ### just labeled

        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]

        poses = np.array(poses)[np.argsort(np.array(self.body['time'])[poses])[self.last_pos:]]
        negs = np.array(negs)[np.argsort(np.array(self.body['time'])[negs])[self.last_neg:]]

        ###############################################

        # prob = clf.predict_proba(self.csr_mat)[:,:1]
        prob1 = clf.decision_function(self.csr_mat)
        prob = np.array([[x] for x in prob1])


        y = np.array([1 if x == 'yes' else 0 for x in self.body['code']])
        y0 = np.copy(y)

        if len(poses) and reuse:
            all = list(set(poses) | set(negs) | set(self.pool))
        else:
            all = range(len(y))
        ####
        # overweight=1
        # all=overweight*(list(poses)+list(negs))+all
        ####
        ####
        ####

        ##
        # C = Counter(y[all])[1] / num_neg
        # es = linear_model.LogisticRegression(penalty='l2', fit_intercept=True, C=C)
        #
        # es.fit(prob[all], y[all])
        # pos_at = list(es.classes_).index(1)
        #
        # pre0 = es.predict_proba(prob[self.pool])[:, pos_at]
        ##


        pos_num_last = Counter(y0)[1]

        lifes = 1
        life = lifes
        pos_num = Counter(y0)[1]

        while (True):
            C = Counter(y[all])[1]/ (num_neg)
            es = linear_model.LogisticRegression(penalty='l2', fit_intercept=True, C=C)

            es.fit(prob[all], y[all])
            pos_at = list(es.classes_).index(1)


            pre = es.predict_proba(prob[self.pool])[:, pos_at]

            # pre= pre0*0.1+pre*0.9
            # pre0=pre

            y = np.copy(y0)

            sample = prob_sample(pre)
            for x in self.pool[sample]:
                y[x] = 1

            # for x in self.pool[np.argsort(pre)[-int(sum(pre)):]]:
            #     y[x]=1

            pos_num = Counter(y)[1]
            # crit=(pos_num-Counter(y0)[1])/pos_num
            # crit = (pos_num - pos_num_last) / pos_num_last
            # if crit<0.05:
            #     break
            if pos_num == pos_num_last:
                life = life - 1
                if life == 0:
                    break
            else:
                life = lifes
            pos_num_last = pos_num
        esty = pos_num - self.last_pos
        pre = es.predict_proba(prob)[:, pos_at]
        self.record_est['x'].append(len(poses)+len(negs)-self.last_pos-self.last_neg)
        self.record_est['semi'].append(esty)


        return esty, pre


    def estimate_curve2(self, clf, reuse=False, num_neg=0):
        from sklearn import linear_model
        import random


        def prob_sample(probs):
            order = np.argsort(probs)[::-1]
            count = 0
            can = []
            sample = []
            for i, x in enumerate(probs[order]):
                count = count + x
                can.append(order[i])
                if count >= 1:
                    # sample.append(np.random.choice(can,1)[0])
                    sample.append(can[0])
                    count = 0
                    can = []
            return sample

        # def prob_sample(probs):
        #     order = np.argsort(probs)[::-1]
        #     count = 0
        #     can = []
        #     sample = []
        #     where = 1
        #     for i, x in enumerate(probs[order]):
        #         count = count + x
        #         can.append(order[i])
        #         if count >= where:
        #             # sample.append(np.random.choice(can,1)[0])
        #             sample.append(can[0])
        #             where = where + 1
        #             can = []
        #     return sample

        # def prob_sample(probs):
        #     sample=[]
        #     for i,x in enumerate(probs):
        #         if random.random<x:
        #             sample.append(i)
        #     return sample

        ### just labeled

        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]

        poses = np.array(poses)[np.argsort(np.array(self.body['time'])[poses])[self.last_pos:]]
        negs = np.array(negs)[np.argsort(np.array(self.body['time'])[negs])[self.last_neg:]]

        ###############################################

        # prob = clf.predict_proba(self.csr_mat)[:,:1]
        prob1 = clf.decision_function(self.csr_mat)
        prob = np.array([[x] for x in prob1])


        y = np.array([1 if x == 'yes' else 0 for x in self.body['code']])
        y0 = np.copy(y)

        if len(poses) and reuse:
            all = list(set(poses) | set(negs) | set(self.pool))
        else:
            all = range(len(y))


        pos_num_last = Counter(y0)[1]

        lifes = 1
        life = lifes
        pos_num = Counter(y0)[1]

        while (True):
            C = Counter(y[all])[1]/ (num_neg)
            es = linear_model.LogisticRegression(penalty='l2', fit_intercept=True, C=C)

            es.fit(prob[all], y[all])
            pos_at = list(es.classes_).index(1)


            pre = es.predict_proba(prob[self.pool])[:, pos_at]

            # pre= pre0*0.1+pre*0.9
            # pre0=pre

            y = np.copy(y0)

            sample = prob_sample(pre)
            for x in self.pool[sample]:
                y[x] = 1

            # for x in self.pool[np.argsort(pre)[-int(sum(pre)):]]:
            #     y[x]=1

            pos_num = Counter(y)[1]
            # crit=(pos_num-Counter(y0)[1])/pos_num
            # crit = (pos_num - pos_num_last) / pos_num_last
            # if crit<0.05:
            #     break
            if pos_num == pos_num_last:
                life = life - 1
                if life == 0:
                    break
            else:
                life = lifes
            pos_num_last = pos_num
        esty = pos_num - self.last_pos
        pre = es.predict_proba(prob)[:, pos_at]
        self.record_est['x'].append(len(poses)+len(negs)-self.last_pos-self.last_neg)
        self.record_est['semi'].append(esty)


        return esty, pre

    def estimate_curve3(self, clf, reuse=False, num_neg=0):
        from sklearn import linear_model
        import random


        # def prob_sample(probs):
        #     order = np.argsort(probs)[::-1]
        #     count = 0
        #     can = []
        #     sample = []
        #     for i, x in enumerate(probs[order]):
        #         count = count + x
        #         can.append(order[i])
        #         if count >= 1:
        #             # sample.append(np.random.choice(can,1)[0])
        #             sample.append(can[0])
        #             count = 0
        #             can = []
        #     return sample

        # def prob_sample(probs):
        #     order = np.argsort(probs)[::-1]
        #     count = 0
        #     can = []
        #     sample = []
        #     where = 1
        #     for i, x in enumerate(probs[order]):
        #         count = count + x
        #         can.append(order[i])
        #         if count >= where:
        #             # sample.append(np.random.choice(can,1)[0])
        #             sample.append(can[0])
        #             where = where + 1
        #             can = []
        #     return sample

        def prob_sample(probs):
            sample=[]
            for i,x in enumerate(probs):
                if random.random<x:
                    sample.append(i)
            return sample

        ### just labeled

        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]

        poses = np.array(poses)[np.argsort(np.array(self.body['time'])[poses])[self.last_pos:]]
        negs = np.array(negs)[np.argsort(np.array(self.body['time'])[negs])[self.last_neg:]]

        ###############################################

        # prob = clf.predict_proba(self.csr_mat)[:,:1]
        prob1 = clf.decision_function(self.csr_mat)
        prob = np.array([[x] for x in prob1])


        y = np.array([1 if x == 'yes' else 0 for x in self.body['code']])
        y0 = np.copy(y)

        if len(poses) and reuse:
            all = list(set(poses) | set(negs) | set(self.pool))
        else:
            all = range(len(y))


        pos_num_last = Counter(y0)[1]

        lifes = 1
        life = lifes
        pos_num = Counter(y0)[1]

        while (True):
            C = Counter(y[all])[1]/ (num_neg)
            es = linear_model.LogisticRegression(penalty='l2', fit_intercept=True, C=C)

            es.fit(prob[all], y[all])
            pos_at = list(es.classes_).index(1)


            pre = es.predict_proba(prob[self.pool])[:, pos_at]

            # pre= pre0*0.1+pre*0.9
            # pre0=pre

            y = np.copy(y0)

            sample = prob_sample(pre)
            for x in self.pool[sample]:
                y[x] = 1

            # for x in self.pool[np.argsort(pre)[-int(sum(pre)):]]:
            #     y[x]=1

            pos_num = Counter(y)[1]
            # crit=(pos_num-Counter(y0)[1])/pos_num
            # crit = (pos_num - pos_num_last) / pos_num_last
            # if crit<0.05:
            #     break
            if pos_num == pos_num_last:
                life = life - 1
                if life == 0:
                    break
            else:
                life = lifes
            pos_num_last = pos_num
        esty = pos_num - self.last_pos
        pre = es.predict_proba(prob)[:, pos_at]
        self.record_est['x'].append(len(poses)+len(negs)-self.last_pos-self.last_neg)
        self.record_est['semi'].append(esty)


        return esty, pre

    def estimate_curve4(self, clf, reuse=False, num_neg=0):
        from sklearn import linear_model
        import random


        def prob_sample(probs):
            order = np.argsort(probs)[::-1]
            count = 0
            can = []
            sample = []
            for i, x in enumerate(probs[order]):
                count = count + x
                can.append(order[i])
                if count >= 1:
                    # sample.append(np.random.choice(can,1)[0])
                    sample.append(can[0])
                    count = 0
                    can = []
            return sample

        # def prob_sample(probs):
        #     order = np.argsort(probs)[::-1]
        #     count = 0
        #     can = []
        #     sample = []
        #     where = 1
        #     for i, x in enumerate(probs[order]):
        #         count = count + x
        #         can.append(order[i])
        #         if count >= where:
        #             # sample.append(np.random.choice(can,1)[0])
        #             sample.append(can[0])
        #             where = where + 1
        #             can = []
        #     return sample

        # def prob_sample(probs):
        #     sample=[]
        #     for i,x in enumerate(probs):
        #         if random.random<x:
        #             sample.append(i)
        #     return sample

        ### just labeled

        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]

        poses = np.array(poses)[np.argsort(np.array(self.body['time'])[poses])[self.last_pos:]]
        negs = np.array(negs)[np.argsort(np.array(self.body['time'])[negs])[self.last_neg:]]

        ###############################################

        # prob = clf.predict_proba(self.csr_mat)[:,:1]
        prob1 = clf.decision_function(self.csr_mat)
        prob = np.array([[x] for x in prob1])


        y = np.array([1 if x == 'yes' else 0 for x in self.body['code']])
        y0 = np.copy(y)

        if len(poses) and reuse:
            all = list(set(poses) | set(negs) | set(self.pool))
        else:
            all = range(len(y))


        pos_num_last = Counter(y0)[1]

        lifes = 1
        life = lifes
        pos_num = Counter(y0)[1]
        C = Counter(y[all])[1]/ (num_neg)

        while (True):

            es = linear_model.LogisticRegression(penalty='l2', fit_intercept=True, C=C)

            es.fit(prob[all], y[all])
            pos_at = list(es.classes_).index(1)


            pre = es.predict_proba(prob[self.pool])[:, pos_at]

            # pre= pre0*0.1+pre*0.9
            # pre0=pre

            y = np.copy(y0)

            sample = prob_sample(pre)
            for x in self.pool[sample]:
                y[x] = 1

            # for x in self.pool[np.argsort(pre)[-int(sum(pre)):]]:
            #     y[x]=1

            pos_num = Counter(y)[1]
            # crit=(pos_num-Counter(y0)[1])/pos_num
            # crit = (pos_num - pos_num_last) / pos_num_last
            # if crit<0.05:
            #     break
            if pos_num == pos_num_last:
                life = life - 1
                if life == 0:
                    break
            else:
                life = lifes
            pos_num_last = pos_num
        esty = pos_num - self.last_pos
        pre = es.predict_proba(prob)[:, pos_at]
        self.record_est['x'].append(len(poses)+len(negs)-self.last_pos-self.last_neg)
        self.record_est['semi'].append(esty)


        return esty, pre

    ## Train model ##
    def train(self,pne=True,weighting=True):

        clf = svm.SVC(kernel='linear', probability=True, class_weight='balanced') if weighting else svm.SVC(kernel='linear', probability=True)



        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]
        left = poses
        decayed = list(left) + list(negs)
        unlabeled = np.where(np.array(self.body['code']) == "undetermined")[0]
        try:
            unlabeled = np.random.choice(unlabeled,size=np.max((len(left),self.atleast)),replace=False)
        except:
            pass

        if not pne:
            unlabeled=[]

        labels=np.array([x if x!='undetermined' else 'no' for x in self.body['code']])
        all_neg=list(negs)+list(unlabeled)
        sample = list(decayed) + list(unlabeled)

        clf.fit(self.csr_mat[sample], labels[sample])


        ## aggressive undersampling ##
        if len(poses)>=self.enough:

            train_dist = clf.decision_function(self.csr_mat[all_neg])
            pos_at = list(clf.classes_).index("yes")
            if pos_at:
                train_dist=-train_dist
            negs_sel = np.argsort(train_dist)[::-1][:len(left)]
            sample = list(left) + list(np.array(all_neg)[negs_sel])
            clf.fit(self.csr_mat[sample], labels[sample])
        elif pne:
            train_dist = clf.decision_function(self.csr_mat[unlabeled])
            pos_at = list(clf.classes_).index("yes")
            if pos_at:
                train_dist = -train_dist
            unlabel_sel = np.argsort(train_dist)[::-1][:int(len(unlabeled) / 2)]
            sample = list(decayed) + list(np.array(unlabeled)[unlabel_sel])
            clf.fit(self.csr_mat[sample], labels[sample])

        ## correct errors with human-machine disagreements ##
        if self.round==self.interval:
            self.round=0
            if self.correction=='knee':
                susp, conf = self.knee_error(clf)
                return susp, conf, susp, conf
            elif self.correction=='machine' or self.correction=='random':
                susp, conf = self.susp(clf)
                return susp, conf, susp, conf

        else:
            self.round = self.round + 1
        #####################################################


        uncertain_id, uncertain_prob = self.uncertain(clf)
        certain_id, certain_prob = self.certain(clf)
        if self.enable_est:
            if self.last_pos>0 and len(poses)-self.last_pos>0:
                self.est_num, self.est = self.estimate_curve(clf, reuse=True, num_neg=len(sample)-len(left))
            else:
                # self.est_num, self.est = self.estimate_curve4(clf, reuse=False, num_neg=len(sample)-len(left))
                # print(self.est_num)
                #
                # self.est_num, self.est = self.estimate_curve(clf, reuse=False, num_neg=len(sample)-len(left))
                # print(self.est_num)
                self.est_num, self.est = self.estimate_curve2(clf, reuse=False, num_neg=len(sample)-len(left))
                # print(self.est_num)
            return uncertain_id, self.est[uncertain_id], certain_id, self.est[certain_id]
        else:
            return uncertain_id, uncertain_prob, certain_id, certain_prob

    ## reuse
    def train_reuse(self,pne=True):
        pne=True
        clf = svm.SVC(kernel='linear', probability=True)
        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]

        left = np.array(poses)[np.argsort(np.array(self.body['time'])[poses])[self.last_pos:]]
        negs = np.array(negs)[np.argsort(np.array(self.body['time'])[negs])[self.last_neg:]]

        if len(left)==0:
            return [], [], self.random(), []



        decayed = list(left) + list(negs)

        unlabeled = np.where(np.array(self.body['code']) == "undetermined")[0]
        try:
            unlabeled = np.random.choice(unlabeled, size=np.max((len(decayed), self.atleast)), replace=False)
        except:
            pass

        if not pne:
            unlabeled = []


        labels = np.array([x if x != 'undetermined' else 'no' for x in self.body['code']])
        all_neg = list(negs) + list(unlabeled)
        sample = list(decayed) + list(unlabeled)

        clf.fit(self.csr_mat[sample], labels[sample])
        ## aggressive undersampling ##
        if len(poses) >= self.enough:
            train_dist = clf.decision_function(self.csr_mat[all_neg])
            pos_at = list(clf.classes_).index("yes")
            if pos_at:
                train_dist=-train_dist
            negs_sel = np.argsort(train_dist)[::-1][:len(left)]
            sample = list(left) + list(np.array(all_neg)[negs_sel])
            clf.fit(self.csr_mat[sample], labels[sample])
        elif pne:
            train_dist = clf.decision_function(self.csr_mat[unlabeled])
            pos_at = list(clf.classes_).index("yes")
            if pos_at:
                train_dist = -train_dist
            unlabel_sel = np.argsort(train_dist)[::-1][:int(len(unlabeled) / 2)]
            sample = list(decayed) + list(np.array(unlabeled)[unlabel_sel])
            clf.fit(self.csr_mat[sample], labels[sample])

        uncertain_id, uncertain_prob = self.uncertain(clf)
        certain_id, certain_prob = self.certain(clf)

        if self.enable_est:
            self.est_num, self.est = self.estimate_curve(clf, reuse=False, num_neg=len(sample)-len(left))
            return uncertain_id, self.est[uncertain_id], certain_id, self.est[certain_id]
        else:
            return uncertain_id, uncertain_prob, certain_id, certain_prob




    ## Get certain ##
    def certain(self,clf):
        pos_at = list(clf.classes_).index("yes")
        prob = clf.predict_proba(self.csr_mat[self.pool])[:,pos_at]
        order = np.argsort(prob)[::-1][:self.step]
        return np.array(self.pool)[order],np.array(prob)[order]

    ## Get uncertain ##
    def uncertain(self,clf):
        pos_at = list(clf.classes_).index("yes")
        prob = clf.predict_proba(self.csr_mat[self.pool])[:, pos_at]
        train_dist = clf.decision_function(self.csr_mat[self.pool])
        order = np.argsort(np.abs(train_dist))[:self.step]  ## uncertainty sampling by distance to decision plane
        # order = np.argsort(np.abs(prob-0.5))[:self.step]    ## uncertainty sampling by prediction probability
        return np.array(self.pool)[order], np.array(prob)[order]

    ## Get random ##
    def random(self):
        return np.random.choice(self.pool,size=np.min((self.step,len(self.pool))),replace=False)

    ## Get one random ##
    def one_rand(self):
        pool_yes = filter(lambda x: self.body['label'][x]=='yes', range(len(self.body['label'])))
        return np.random.choice(pool_yes, size=1, replace=False)

    ## Format ##
    def format(self,id,prob=[]):
        result=[]
        for ind,i in enumerate(id):
            tmp = {key: self.body[key][i] for key in self.body}
            tmp["id"]=str(i)
            if prob!=[]:
                tmp["prob"]=prob[ind]
            result.append(tmp)
        return result

    ## Code candidate studies ##
    def code(self,id,label):
        self.body["code"][id] = label
        self.body["time"][id] = time.time()

    def code_error(self,id,error='none'):
        if error=='circle':
            self.code_circle(id, self.body['label'][id])
        elif error=='random':
            self.code_random(id, self.body['label'][id])
        elif error=='random3':
            self.code_random3(id, self.body['label'][id])
        elif error=='three':
            self.code_three(id, self.body['label'][id])
        else:
            self.code(id, self.body['label'][id])

    def code_circle(self,id,label):
        import random
        if random.random()<0.0:
            self.body["code"][id] = label
        else:
            self.body["code"][id] = 'yes' if random.random()<float(self.body['syn_error'][id]) else 'no'
        self.body["time"][id] = time.time()

    def code_three(self, id, label):
        a = self.code_random(id,label)
        b = self.code_random(id,label)
        if a == 'yes' or b=='yes':
            self.body["code"][id] = 'yes'

    def code_random(self,id,label):
        import random

        if label=='yes':
            if random.random()<self.false_neg:
                new = 'no'
            else:
                new = 'yes'
        else:
            # if random.random()<self.false_alarm:
            #     new = 'yes'
            # else:
            new = 'no'
        if new == 'yes' or self.body["count"][id]>0:
            self.body['fixed'][id]=1
        self.body["code"][id] = new
        self.body["time"][id] = time.time()
        self.body["count"][id] = self.body["count"][id] + 1
        return new

    def code_random3(self,id,label):
        import random

        if label=='yes':
            if self.body["count"][id]>0:
                if random.random()<self.false_neg**2:
                    new = 'no'
                else:
                    new = 'yes'
            else:
                if random.random()<self.false_neg:
                    new = 'no'
                else:
                    new = 'yes'
        else:
            # if random.random()<self.false_alarm:
            #     new = 'yes'
            # else:
            new = 'no'

        if new == 'yes' or self.body["count"][id]>0:
            self.body['fixed'][id]=1
            self.body["count"][id] = self.body["count"][id] + 1

        # if new == 'yes' or self.body["count"][id]==2:
        #
        #     self.body['fixed'][id]=1
        # elif self.body["count"][id] == 1:
        #     self.body["count"][id] = self.body["count"][id] + 1
        #     self.code_random3(id,label)
        self.body["code"][id] = new
        self.body["time"][id] = time.time()
        self.body["count"][id] = self.body["count"][id] + 1



    ## Get suspecious codes
    def susp(self,clf):
        thres_pos = 1
        thres_neg = 1
        length_pos = 0
        length_neg = int(self.neg_len*self.step)


        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]
        poses = np.array(poses)[np.argsort(np.array(self.body['time'])[poses])[self.last_pos:]]
        negs = np.array(negs)[np.argsort(np.array(self.body['time'])[negs])[self.last_neg:]]

        poses = np.array(poses)[np.where(np.array(self.body['fixed'])[poses] == 0)[0]]
        negs = np.array(negs)[np.where(np.array(self.body['fixed'])[negs] == 0)[0]]

        # length_pos = int(0.02*len(poses))
        # length_neg = int(0.2*len(negs))

        if len(poses)>0:
            pos_at = list(clf.classes_).index("yes")
            prob_pos = clf.predict_proba(self.csr_mat[poses])[:,pos_at]
            se_pos = np.argsort(prob_pos)[:length_pos]
            se_pos = [s for s in se_pos if prob_pos[s]<thres_pos]
            sel_pos = poses[se_pos]
            # print(np.array(self.body['label'])[sel_pos])
            if self.correction=='random':
                sel_pos = np.random.choice(poses,length_pos,replace=False)
        else:
            sel_pos = np.array([])
            # print('null')

        if len(negs)>0:
            neg_at = list(clf.classes_).index("no")
            prob_neg = clf.predict_proba(self.csr_mat[negs])[:,neg_at]
            se_neg = np.argsort(prob_neg)[:length_neg]
            se_neg = [s for s in se_neg if prob_neg[s]<thres_neg]
            sel_neg = negs[se_neg]
            # print(np.array(self.body['label'])[sel_neg])
            if self.correction=='random':
                sel_neg = np.random.choice(negs,length_neg,replace=False)
        else:
            sel_neg = np.array([])
            # print('null')
        try:
            probs = prob_pos[se_pos].tolist() + prob_neg[se_neg].tolist()
        except:
            probs = []
        return sel_pos.tolist() + sel_neg.tolist(), probs

    ## Plot ##
    def plot(self):
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 20}

        plt.rc('font', **font)
        paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
                 'figure.autolayout': True, 'figure.figsize': (16, 8)}

        plt.rcParams.update(paras)

        fig = plt.figure()
        plt.plot(self.record['x'], self.record["pos"])
        ### estimation ####
        if Counter(self.body['code'])['yes']>=self.enough:
            est=self.est2[self.pool]
            order=np.argsort(est)[::-1]
            xx=[self.record["x"][-1]]
            yy=[self.record["pos"][-1]]
            for x in xrange(int(len(order)/self.step)):
                delta = sum(est[order[x*self.step:(x+1)*self.step]])
                if delta>=0.1:
                    yy.append(yy[-1]+delta)
                    xx.append(xx[-1]+self.step)
                else:
                    break
            plt.plot(xx, yy, "-.")
        ####################
        plt.ylabel("Relevant Found")
        plt.xlabel("Documents Reviewed")
        name=self.name+ "_" + str(int(time.time()))+".png"

        dir = "./static/image"
        for file in os.listdir(dir):
            os.remove(os.path.join(dir, file))

        plt.savefig("./static/image/" + name)
        plt.close(fig)
        return name

    def get_allpos(self):
        return len([1 for c in self.body["label"] if c=="yes"])-self.last_pos

    ## Restart ##
    def restart(self):
        try:
            os.remove("./memory/"+self.name+".pickle")
        except:
            pass

    ## Get missed relevant docs ##
    def get_rest(self):
        rest=[x for x in xrange(len(self.body['label'])) if self.body['label'][x]=='yes' and self.body['code'][x]!='yes']
        rests={}
        # fields = ["Document Title", "Abstract", "Year", "PDF Link"]
        fields = ["Document Title"]
        for r in rest:
            rests[r]={}
            for f in fields:
                rests[r][f]=self.body[f][r]
        set_trace()
        return rests

    def cache_est(self):
        est = self.est[self.pool]
        order = np.argsort(est)[::-1]
        xx = [self.record["x"][-1]]
        yy = [self.record["pos"][-1]]
        for x in xrange(int(len(order) / self.step)):
            delta = sum(est[order[x * self.step:(x + 1) * self.step]])
            if delta >= 0.1:
                yy.append(yy[-1] + delta)
                xx.append(xx[-1] + self.step)
            else:
                break
        self.xx=xx
        self.yy=yy

        est = self.est2[self.pool]
        order = np.argsort(est)[::-1]
        xx2 = [self.record["x"][-1]]
        yy2 = [self.record["pos"][-1]]
        for x in xrange(int(len(order) / self.step)):
            delta = sum(est[order[x * self.step:(x + 1) * self.step]])
            if delta >= 0.1:
                yy2.append(yy2[-1] + delta)
                xx2.append(xx2[-1] + self.step)
            else:
                break
        self.xx2 = xx2
        self.yy2 = yy2
