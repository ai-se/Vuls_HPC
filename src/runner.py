from __future__ import division, print_function


import numpy as np
from pdb import set_trace
from demos import cmd
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sk import rdivDemo

from sklearn import svm
from collections import Counter

from mar import MAR
import pandas as pd





def colorcode(N):
    jet = plt.get_cmap('jet')
    cNorm  = colors.Normalize(vmin=0, vmax=N-1, clip=True)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    return scalarMap





"normalization_row"
def normalize(mat,ord=2):
    mat=mat.asfptype()
    for i in xrange(mat.shape[0]):
        nor=np.linalg.norm(mat[i].data,ord=ord)
        if not nor==0:
            for k in mat[i].indices:
                mat[i,k]=mat[i,k]/nor
    return mat


def hcca_lda(csr_mat,csr_lda, labels, step=10 ,initial=10, pos_limit=1, thres=30, stop=0.9):
    num=len(labels)
    pool=range(num)
    train=[]
    steps = np.array(range(int(num / step))) * step

    pos=0
    pos_track=[0]
    clf = svm.SVC(kernel='linear', probability=True)
    begin=0
    result={}
    enough=False

    total=Counter(labels)["yes"]*stop

    # total = 1000

    for idx, round in enumerate(steps[:-1]):

        if round >= 2500:
            if enough:
                pos_track_f=pos_track9
                train_f=train9
                pos_track_l=pos_track8
                train_l=train8
            elif begin:
                pos_track_f=pos_track4
                train_f=train4
                pos_track_l=pos_track2
                train_l=train2
            else:
                pos_track_f=pos_track
                train_f=train
                pos_track_l=pos_track
                train_l=train
            break

        can = np.random.choice(pool, step, replace=False)
        train.extend(can)
        pool = list(set(pool) - set(can))
        try:
            pos = Counter(labels[train])["yes"]
        except:
            pos = 0
        pos_track.append(pos)

        if not begin:
            pool2=pool[:]
            train2=train[:]
            pos_track2=pos_track[:]
            pool4 = pool2[:]
            train4 = train2[:]
            pos_track4 = pos_track2[:]
            if round >= initial and pos>=pos_limit:
                begin=idx+1
        else:
            clf.fit(csr_mat[train4], labels[train4])
            pred_proba4 = clf.predict_proba(csr_mat[pool4])
            pos_at = list(clf.classes_).index("yes")
            proba4 = pred_proba4[:, pos_at]
            sort_order_certain4 = np.argsort(1 - proba4)
            can4 = [pool4[i] for i in sort_order_certain4[:step]]
            train4.extend(can4)
            pool4 = list(set(pool4) - set(can4))
            pos = Counter(labels[train4])["yes"]
            pos_track4.append(pos)

            ## lda
            clf.fit(csr_lda[train2], labels[train2])
            pred_proba2 = clf.predict_proba(csr_lda[pool2])
            pos_at = list(clf.classes_).index("yes")
            proba2 = pred_proba2[:, pos_at]
            sort_order_certain2 = np.argsort(1 - proba2)
            can2 = [pool2[i] for i in sort_order_certain2[:step]]
            train2.extend(can2)
            pool2 = list(set(pool2) - set(can2))
            pos = Counter(labels[train2])["yes"]
            pos_track2.append(pos)


            ################ new *_C_C_A
            if not enough:
                if pos>=thres:
                    enough=True
                    pos_track9=pos_track4[:]
                    train9=train4[:]
                    pool9=pool4[:]
                    pos_track8=pos_track2[:]
                    train8=train2[:]
                    pool8=pool2[:]
            else:
                clf.fit(csr_mat[train9], labels[train9])
                poses = np.where(labels[train9] == "yes")[0]
                negs = np.where(labels[train9] == "no")[0]
                train_dist = clf.decision_function(csr_mat[train9][negs])
                negs_sel = np.argsort(np.abs(train_dist))[::-1][:len(poses)]
                sample9 = np.array(train9)[poses].tolist() + np.array(train9)[negs][negs_sel].tolist()

                clf.fit(csr_mat[sample9], labels[sample9])
                pred_proba9 = clf.predict_proba(csr_mat[pool9])
                pos_at = list(clf.classes_).index("yes")
                proba9 = pred_proba9[:, pos_at]
                sort_order_certain9 = np.argsort(1 - proba9)
                can9 = [pool9[i] for i in sort_order_certain9[:step]]
                train9.extend(can9)
                pool9 = list(set(pool9) - set(can9))
                pos = Counter(labels[train9])["yes"]
                pos_track9.append(pos)

                clf.fit(csr_lda[train8], labels[train8])
                poses = np.where(labels[train8] == "yes")[0]
                negs = np.where(labels[train8] == "no")[0]
                train_dist = clf.decision_function(csr_lda[train8][negs])
                negs_sel = np.argsort(np.abs(train_dist))[::-1][:len(poses)]
                sample8 = np.array(train8)[poses].tolist() + np.array(train8)[negs][negs_sel].tolist()

                clf.fit(csr_lda[sample8], labels[sample8])
                pred_proba8 = clf.predict_proba(csr_lda[pool8])
                pos_at = list(clf.classes_).index("yes")
                proba8 = pred_proba8[:, pos_at]
                sort_order_certain8 = np.argsort(1 - proba8)
                can8 = [pool8[i] for i in sort_order_certain8[:step]]
                train8.extend(can8)
                pool8 = list(set(pool8) - set(can8))
                pos = Counter(labels[train8])["yes"]
                pos_track8.append(pos)

        print("Round #{id} passed\r".format(id=round), end="")

    result["begin"] = begin
    result["x"] = steps[:len(pos_track_f)]
    result["new_continuous_aggressive"] = pos_track_f
    result["lda"] = pos_track_l
    return result, train_f



##### draw

def use_or_not(file):
    font = {'family': 'cursive',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    with open("../dump/"+str(file)+".pickle", "r") as f:
        results=pickle.load(f)
    with open("../dump/"+str(file)+"0.pickle", "r") as f:
        results0=pickle.load(f)


    stats=bestNworst(results)
    stats0 = bestNworst(results0)
    colors=['blue','purple','green','brown','red']
    lines=['-','--','-.',':']
    five=['best','$Q_1$','median','$Q_3$','worst']

    plt.figure()
    for key in stats0:
        a = key.split("_")[0]
        if a=="start":
            for j,ind in enumerate(stats0[key]):
                plt.plot(stats0[key][ind]['x'], stats0[key][ind]['pos'],linestyle=lines[0],color=colors[j],label=five[j]+"_no")
    for key in stats:
        a = key.split("_")[0]
        if a=="start":
            for j,ind in enumerate(stats[key]):
                plt.plot(stats[key][ind]['x'], stats[key][ind]['pos'],linestyle=lines[1],color=colors[j],label=five[j]+"_yes")


    plt.ylabel("Retrieval Rate")
    plt.xlabel("Studies Reviewed")
    plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=2, borderaxespad=0.)
    plt.savefig("../figure/"+str(file).split('_')[1]+".eps")
    plt.savefig("../figure/"+str(file).split('_')[1]+".png")

def stats(file):

    with open("../dump/"+str(file)+".pickle", "r") as f:
        results=pickle.load(f)
    name=file.split('_')[0]
    test=[]
    for key in results:
        a=key.split('_')[0]
        try:
            b=key.split('_')[1]
        except:
            b='0'
        if b=="2" and name=="UPDATE":
            if a=="POS":
                a="UPDATE_POS"
            elif a=="UPDATE":
                a="UPDATE_ALL"
            tmp=[]
            for r in results[key]:
                tmp.append(r['x'][-1])
            print(a+": max %d" %max(tmp))
            test.append([a]+tmp)
        elif name!="UPDATE":
            tmp=[]
            for r in results[key]:
                tmp.append(r['x'][-1])
            test.append([a]+tmp)
            print(a+": max %d" %max(tmp))
    rdivDemo(test,isLatex=True)
    set_trace()


def analyze(read):
    unknown = np.where(np.array(read.body['code']) == "undetermined")[0]
    pos = np.where(np.array(read.body['code']) == "yes")[0]
    neg = np.where(np.array(read.body['code']) == "no")[0]
    yes = np.where(np.array(read.body['label']) == "yes")[0]
    no = np.where(np.array(read.body['label']) == "no")[0]
    falsepos = len(set(pos) & set(no))
    truepos = len(set(pos) & set(yes))
    falseneg = len(set(neg) & set(yes))
    unknownyes = len(set(unknown) & set(yes))
    unique = len(read.body['code']) - len(unknown)
    count = sum(read.body['count'])
    correction = read.correction
    return {"falsepos": falsepos, "truepos": truepos, "falseneg": falseneg, "unknownyes": unknownyes, "unique": unique, "count": count, "correction": correction}



def draw(file):
    font = {'family': 'cursive',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    with open("../dump/"+str(file)+".pickle", "r") as f:
        results=pickle.load(f)


    stats=bestNworst(results)
    colors=['blue','purple','green','brown','red']
    lines=['-','--','-.',':']
    five=['best','$Q_1$','median','$Q_3$','worst']

    nums = set([])
    line=[0,0,0,0,0]

    for key in stats:
        a = key.split("_")[0]
        if a=="start":
            a='FASTREAD'
        if a=="POS":
            a='UPDATE_POS'
        if a=="UPDATE":
            a='UPDATE_ALL'
        try:
            b = key.split("_")[1]
        except:
            b = 0
        nums = nums | set([b])
        plt.figure(int(b))
        for j,ind in enumerate(stats[key]):
            if ind == 50 or ind == 100:
                plt.plot(stats[key][ind]['x'], stats[key][ind]['pos'],linestyle=lines[line[int(b)]],color=colors[j],label=five[j]+"_"+str(a).capitalize())
        line[int(b)]+=1

    for i in nums:
        plt.figure(int(i))
        plt.ylabel(str(file).split("_")[1]+"\nRelevant Studies")
        plt.xlabel("Studies Reviewed")
        plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=2, borderaxespad=0.)
        plt.savefig("../figure/"+str(file)+str(i)+".eps")
        plt.savefig("../figure/"+str(file)+str(i)+".png")


def draw_soa(file):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    with open("../dump/"+str(file)+".pickle", "r") as f:
        results=pickle.load(f)


    stats=bestNworst(results)
    colors=['blue','purple','green','brown','red']
    lines=['-','--','-.',':']
    five=['best','$Q_1$','median','$Q_3$','worst']

    nums = set([])
    line=[0,0,0,0,0]

    xmax = stats['LINEAR'][100]['x'][-1]
    ymax = stats['LINEAR'][100]['pos'][-1]
    for key in stats:
        a = key.split("_")[0]
        try:
            b = key.split("_")[1]
        except:
            b = 0
        if key=="LOC":
            continue
        if key=="CART":
            continue

        nums = nums | set([b])
        plt.figure(int(b))
        for j,ind in enumerate(stats[key]):
            if ind == 50:
                plt.plot(np.array(stats[key][ind]['x'])/xmax, np.array(stats[key][ind]['pos'])/ymax,linestyle=lines[line[int(b)]],label=str(a))
                # plt.plot(stats[key][ind]['x'], stats[key][ind]['pos'],label=str(a))
        line[int(b)]+=1

    for i in nums:
        plt.figure(int(i))
        plt.plot(0.163,0.49,color="black",marker='o')
        plt.ylabel("Recall")
        plt.xlabel("Cost")
        plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=1, borderaxespad=0.)
        plt.savefig("../figure/"+str(file)+str(i)+".eps")
        plt.savefig("../figure/"+str(file)+str(i)+".png")


def draw_trans(file):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    with open("../dump/soa_"+str(file)+".pickle", "r") as f:
        results=pickle.load(f)
    with open("../dump/trans_"+str(file)+".pickle", "r") as f:
        results1=pickle.load(f)


    stats=bestNworst(results)
    stats1 = bestNworst(results1)
    colors=['blue','purple','green','brown','red']
    lines=['-','--','-.',':']
    five=['best','$Q_1$','median','$Q_3$','worst']

    nums = set([])
    line=[0,0,0,0,0]

    xmax = stats['LINEAR'][100]['x'][-1]
    ymax = stats['LINEAR'][100]['pos'][-1]
    b=0

    for key in stats:
        if key=="LOC":
            continue
        for j,ind in enumerate(stats[key]):
            if ind == 50:
                plt.plot(np.array(stats[key][ind]['x'])/xmax, np.array(stats[key][ind]['pos'])/ymax,linestyle=lines[line[int(b)]],label=str(key))
                # plt.plot(stats[key][ind]['x'], stats[key][ind]['pos'],label=str(a))
        line[int(b)]+=1
    for key in stats1:
        if key!="RF":
            continue
        for j,ind in enumerate(stats1[key]):
            if ind == 50:
                plt.plot(np.array(stats1[key][ind]['x'])/xmax, np.array(stats1[key][ind]['pos'])/ymax,linestyle=lines[line[int(b)]],label='trans_'+str(key))
                # plt.plot(stats[key][ind]['x'], stats[key][ind]['pos'],label=str(a))
        line[int(b)]+=1

    i=1
    plt.plot(0.163, 0.49, color="black", marker='o')
    plt.ylabel("Recall")
    plt.xlabel("Cost")
    plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/trans_"+str(file)+str(i)+".eps")
    plt.savefig("../figure/trans_"+str(file)+str(i)+".png")

def draw_pre(file,what="c"):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    with open("../dump/"+str(file)+".pickle", "r") as f:
        results=pickle.load(f)


    stats=bestNworst(results)
    colors=['blue','purple','green','brown','red']
    lines=['-','--','-.',':']
    five=['best','$Q_1$','median','$Q_3$','worst']

    nums = set([])
    line=[0,0,0,0,0]

    for key in stats:
        a = key.split("_")[0]
        if a=="start":
            a='MAT'
        if a=="FASTREAD":
            a='MAT'
        if a=="POS":
            a='UPDATE_POS'
        if a=="UPDATE":
            a='UPDATE_ALL'
        try:
            b = key.split("_")[1]
        except:
            b = 0

        if a=="LINEAR":
            continue

        if what=="c":
            if a=="SVM-rbf" or a=="SVM-sigmoid" or a=="SVM-poly":
                continue
            if a=="SVM-linear":
                a='SVM'
        else:
            if a!="SVM-rbf" and a!="SVM-sigmoid" and a!="SVM-linear" and a!="SVM-poly":
                continue



        nums = nums | set([b])
        plt.figure(int(b))
        for j,ind in enumerate(stats[key]):
            if ind == 50:
                plt.plot(stats[key][ind]['x'], stats[key][ind]['pos'],linestyle=lines[line[int(b)]],label=str(a))
                # plt.plot(stats[key][ind]['x'], stats[key][ind]['pos'],label=str(a))
        line[int(b)]+=1

    for i in nums:
        plt.figure(int(i))
        plt.ylabel("Bugs Found")
        plt.xlabel("Modules Tested")
        plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=1, borderaxespad=0.)
        plt.savefig("../figure/"+what+"_"+str(file)+str(i)+".eps")
        plt.savefig("../figure/"+what+"_"+str(file)+str(i)+".png")

def update_median_draw(file):
    font = {'family': 'cursive',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    with open("../dump/"+str(file)+".pickle", "r") as f:
        results=pickle.load(f)


    stats=bestNworst(results)
    colors=['blue','purple','green','brown','red']
    lines=['-','--','-.',':']
    five=['best','$Q_1$','median','$Q_3$','worst']

    nums = set([])
    line=[0,0,0,0,0]
    for key in stats:
        a = key.split("_")[0]
        if a=="start":
            a='FASTREAD'
        try:
            b = key.split("_")[1]
        except:
            b = 0
        nums = nums | set([b])
        plt.figure(int(b))
        for j,ind in enumerate(stats[key]):
            if ind==50:
                plt.plot(stats[key][ind]['x'], stats[key][ind]['pos'],linestyle=lines[line[int(b)]],color=colors[j],label=five[j]+"_"+str(a).capitalize())
        line[int(b)]+=1

    for i in nums:
        plt.figure(int(i))
        plt.ylabel("Retrieval Rate")
        plt.xlabel("Studies Reviewed")
        plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=1, borderaxespad=0.)
        plt.savefig("../figure/median_"+str(file)+str(i)+".eps")
        plt.savefig("../figure/median_"+str(file)+str(i)+".png")

def bestNworst(results):
    stats={}

    for key in results:
        stats[key]={}
        result=results[key]
        order = np.argsort([r['x'][-1] for r in result])
        for ind in [0,25,50,75,100]:
            stats[key][ind]=result[order[int(ind*(len(order)-1)/100)]]

    return stats


#################


def parse_data_new(mode='source', path='../../Datasets/firefox/', file0='vulns.csv', file1='clean_firefox_metrics_security.csv'):
    import pandas as pd
    from ast_func import ast_func
    vulns_types = pd.read_csv('../../Datasets/vulns/' + str(file0))
    types = {}
    pretype = ''
    mapping = {'arbitrary-code': 'Arbitrary Code', 'Code - Code Quality': 'Code Quality', 'Code - Resource Management Error - Improper Resource Shutdown or Release': 'Improper Control of a Resource Through its Lifetime', 'buffer-overflow': 'Range Error', 'data-leakage': 'Improper Control of a Resource Through its Lifetime', 'use-after-free': 'Improper Control of a Resource Through its Lifetime', 'Code - Resource Management Error - Uncontrolled Resource Consumption': 'Improper Control of a Resource Through its Lifetime', 'Code - Time and State - Race Conditions': 'Other', 'memory-corruption': 'Range Error', 'Code - Resource Management Error': 'Improper Control of a Resource Through its Lifetime', 'Code - Traversal - Link Following': 'Other', '?': 'Other', 'spoofing': 'Other', 'privilege-escalation': 'Other', 'Code - Traversal': 'Other', 'Code - Data Processing': 'Other', 'cross-site-scripting': 'Other', 'exploitable-crash': 'Other', 'Environment': 'Other', 'Code - Resource Management Error - Use After Free': 'Improper Control of a Resource Through its Lifetime', 'denial-of-service': 'Improper Control of a Resource Through its Lifetime', 'Configuration': 'Other', 'injection': 'Other', 'Code - Security Features - Protection Mechanism Failure': 'Other'}
    for i, name in enumerate(vulns_types['file']):
        if name not in types:
            types[name] = []
        if not pd.isnull(vulns_types['type'][i]):
            pretype = mapping[vulns_types['type'][i]]
        types[name].append(pretype)

    vulns = pd.read_csv('../../Datasets/vulns/' + str(file1))
    type_col = []
    newcol = []
    for i,name in enumerate(vulns['file']):
        if vulns['severity'][i] == 0:
            type_col.append('')
        else:
            type_col.append(','.join(list(set(types[name]))))
        with open(path+name,'r') as f:
            sourcecode = f.read()
            if not sourcecode:
                sourcecode = name
            if mode=='source':
                newcol.append(sourcecode)
            elif mode=='ast':
                newcol.append(' '.join(ast_func(sourcecode)))
            else:
                newcol.append(sourcecode)

    vulns['sourcecode'] = newcol
    vulns['type'] = type_col
    set_trace()
    vulns.to_csv('../../Datasets/vulns/vuls_data_new.csv')




def divide_type(file='../../Datasets/vulns/vuls_data_new.csv'):
    import pandas as pd
    body = pd.read_csv(file)
    all_types = []
    for types in body['type']:
        if pd.isnull(types):
            continue
        else:
            all_types.extend(types.split(','))
    all_types = Counter(all_types)
    print(all_types)
    set_trace()



def CRASH(type, stop='true', error='none', interval = 100000, seed=0):
    stopat = 1
    starting = 1
    np.random.seed(seed)

    read = MAR()
    read = read.create("vuls_data_new.csv",type)
    thres = Counter(read.body.crashes>0)[True]
    read.interval = interval


    num2 = read.get_allpos()
    target = int(num2 * stopat)

    read.enable_est = True

    while True:
        pos, neg, total = read.get_numbers()
        try:
            print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        except:
            print("%d, %d" %(pos,pos+neg))

        if pos + neg >= total:
            if stop=='knee' and error=='random':
                coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                seq = coded[np.argsort(read.body['time'][coded])]
                part1 = set(seq[:read.kneepoint * read.step]) & set(
                    np.where(np.array(read.body['code']) == "no")[0])
                part2 = set(seq[read.kneepoint * read.step:]) & set(
                    np.where(np.array(read.body['code']) == "yes")[0])
                for id in part1 | part2:
                    read.code_error(id, error=error)
            break

        if pos < starting or pos+neg<thres:
            for id in read.BM25_get():
                read.code_error(id, error=error)
        else:
            break
            # a,b,c,d =read.train(weighting=True,pne=True)
            # if stop == 'est':
            #     if stopat * read.est_num <= pos:
            #         break
            # elif stop == 'soft':
            #     if pos>0 and pos_last==pos:
            #         counter = counter+1
            #     else:
            #         counter=0
            #     pos_last=pos
            #     if counter >=5:
            #         break
            # elif stop == 'knee':
            #     if pos>0:
            #         if read.knee():
            #             if error=='random':
            #                 coded = np.where(np.array(read.body['code']) != "undetermined")[0]
            #                 seq = coded[np.argsort(np.array(read.body['time'])[coded])]
            #                 part1 = set(seq[:read.kneepoint * read.step]) & set(
            #                     np.where(np.array(read.body['code']) == "no")[0])
            #                 part2 = set(seq[read.kneepoint * read.step:]) & set(
            #                     np.where(np.array(read.body['code']) == "yes")[0])
            #                 for id in part1|part2:
            #                     read.code_error(id, error=error)
            #             break
            # elif stop == 'true':
            #     if pos >= target:
            #         break
            # elif stop == 'mix':
            #     if pos >= target and stopat * read.est_num <= pos:
            #         break
            # if pos < read.enough:
            #     for id in a:
            #         read.code_error(id, error=error)
            # else:
            #     for id in c:
            #         read.code_error(id, error=error)
    # read.export()
    # results = analyze(read)
    # print(results)
    result = {}
    result['est'] = (read.record_est)
    result['pos'] = (read.record)
    # with open("../dump/"+type+"_crash.pickle","wb") as handle:
    #     pickle.dump(result,handle)
    return read


def BM25(type, stop='true', error='none', correct = 'no', interval = 100000, seed=0):
    stopat = 1
    thres = 0
    starting = 1
    counter = 0
    pos_last = 0
    np.random.seed(seed)

    read = MAR()
    read.correction=correct
    read.crash='append'
    read = read.create("vuls_data_new.csv",type)

    read.interval = interval

    num2 = read.get_allpos()
    target = int(num2 * stopat)

    read.enable_est = True


    while True:
        pos, neg, total = read.get_numbers()
        # try:
        #     print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        # except:
        #     print("%d, %d" %(pos,pos+neg))

        if pos + neg >= total:
            if (stop=='knee') and error=='random':
                coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                seq = coded[np.argsort(read.body['time'][coded])]
                part1 = set(seq[:read.record['x'][read.kneepoint]]) & set(
                    np.where(np.array(read.body['code']) == "no")[0])
                # part2 = set(seq[read.record['x'][read.kneepoint]:]) & set(
                #     np.where(np.array(read.body['code']) == "yes")[0])
                # for id in part1 | part2:
                for id in part1:
                    read.code_error(id, error=error)
            break

        if pos < starting or pos+neg<thres:
            for id in read.BM25_get():
                read.code_error(id, error=error)
        else:
            a,b,c,d =read.train(weighting=True,pne=True)
            if stop == 'est':
                if stopat * read.est_num <= pos:
                    break
            elif stop == 'soft':
                if pos>0 and pos_last==pos:
                    counter = counter+1
                else:
                    counter=0
                pos_last=pos
                if counter >=5:
                    break
            elif stop == 'knee':
                if pos>0:
                    if read.knee():
                        if error=='random':
                            coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                            seq = coded[np.argsort(np.array(read.body['time'])[coded])]
                            part1 = set(seq[:read.kneepoint * read.step]) & set(
                                np.where(np.array(read.body['code']) == "no")[0])
                            part2 = set(seq[read.kneepoint * read.step:]) & set(
                                np.where(np.array(read.body['code']) == "yes")[0])
                            for id in part1|part2:
                                read.code_error(id, error=error)
                        break
            elif stop == 'true':
                if pos >= target:
                    break
            elif stop == 'mix':
                if pos >= target and stopat * read.est_num <= pos:
                    break
            if pos < 10:
                for id in a:
                    read.code_error(id, error=error)
            else:
                for id in c:
                    read.code_error(id, error=error)
    # read.export()
    results = analyze(read)
    print(results)
    return read


def Metrics(type, stop='true', error='none', interval = 100000, seed=0):
    stopat = 1
    thres = 0
    starting = 1
    counter = 0
    pos_last = 0
    np.random.seed(seed)

    read = MAR()
    read.norm = 'l2col'
    read.metrics='only'

    read = read.create("vuls_data_new.csv",type)
    read.step = 100

    read.interval = interval


    num2 = read.get_allpos()
    target = int(num2 * stopat)

    read.enable_est = False


    while True:
        pos, neg, total = read.get_numbers()
        # try:
        #     print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        # except:
        #     print("%d, %d" %(pos,pos+neg))

        if pos + neg >= total:
            if stop=='knee' and error=='random':
                coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                seq = coded[np.argsort(read.body['time'][coded])]
                part1 = set(seq[:read.kneepoint * read.step]) & set(
                    np.where(np.array(read.body['code']) == "no")[0])
                part2 = set(seq[read.kneepoint * read.step:]) & set(
                    np.where(np.array(read.body['code']) == "yes")[0])
                for id in part1 | part2:
                    read.code_error(id, error=error)
            break

        if pos < starting or pos+neg<thres:
            for id in read.random():
                read.code_error(id, error=error)
        else:
            a,b,c,d =read.train(weighting=True,pne=True)
            if stop == 'est':
                if stopat * read.est_num <= pos:
                    break
            elif stop == 'soft':
                if pos>0 and pos_last==pos:
                    counter = counter+1
                else:
                    counter=0
                pos_last=pos
                if counter >=5:
                    break
            elif stop == 'knee':
                if pos>0:
                    if read.knee():
                        if error=='random':
                            coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                            seq = coded[np.argsort(np.array(read.body['time'])[coded])]
                            part1 = set(seq[:read.kneepoint * read.step]) & set(
                                np.where(np.array(read.body['code']) == "no")[0])
                            part2 = set(seq[read.kneepoint * read.step:]) & set(
                                np.where(np.array(read.body['code']) == "yes")[0])
                            for id in part1|part2:
                                read.code_error(id, error=error)
                        break
            elif stop == 'true':
                if pos >= target:
                    break
            elif stop == 'mix':
                if pos >= target and stopat * read.est_num <= pos:
                    break
            if pos < 10:
                for id in a:
                    read.code_error(id, error=error)
            else:
                for id in c:
                    read.code_error(id, error=error)
    # read.export()
    # results = analyze(read)
    # print(results)
    result = {}
    result['est'] = (read.record_est)
    result['pos'] = (read.record)
    return read

def Random(type, stop='true', error='none', error_rate = 0.5, correct = 'no', interval = 100000, seed=0, neg_len=0.5):
    stopat = 1
    thres = 0
    starting = 1
    counter = 0
    pos_last = 0
    np.random.seed(seed)
    read = MAR()

    read.false_neg = float(error_rate)
    read.correction=correct
    read.neg_len=float(neg_len)
    read = read.create("vuls_data_new.csv",type)

    read.interval = interval

    num2 = read.get_allpos()
    target = int(num2 * stopat)

    read.enable_est = True


    while True:
        pos, neg, total = read.get_numbers()
        # try:
        #     print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        # except:
        #     print("%d, %d" %(pos,pos+neg))

        if pos + neg >= total:
            if (stop=='knee') and error=='random':
                coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                seq = coded[np.argsort(read.body['time'][coded])]
                part1 = set(seq[:read.record['x'][read.kneepoint]]) & set(
                    np.where(np.array(read.body['code']) == "no")[0])
                # part2 = set(seq[read.record['x'][read.kneepoint]:]) & set(
                #     np.where(np.array(read.body['code']) == "yes")[0])
                # for id in part1 | part2:
                for id in part1:
                    read.code_error(id, error=error)
            break

        if pos < starting or pos+neg<thres:
            for id in read.random():
                read.code_error(id, error=error)
        else:
            a,b,c,d =read.train(weighting=True,pne=True)
            if stop == 'est':
                if stopat * read.est_num <= pos:
                    break
            elif stop == 'soft':
                if pos>0 and pos_last==pos:
                    counter = counter+1
                else:
                    counter=0
                pos_last=pos
                if counter >=5:
                    break
            elif stop == 'knee':
                if pos>0:
                    if read.knee():
                        if error=='random':
                            coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                            seq = coded[np.argsort(np.array(read.body['time'])[coded])]
                            part1 = set(seq[:read.kneepoint * read.step]) & set(
                                np.where(np.array(read.body['code']) == "no")[0])
                            # part2 = set(seq[read.kneepoint * read.step:]) & set(
                            #     np.where(np.array(read.body['code']) == "yes")[0])
                            # for id in part1 | part2:
                            for id in part1:
                                read.code_error(id, error=error)
                        break
            elif stop == 'true':
                if pos >= target:
                    break
            elif stop == 'mix':
                if pos >= target and stopat * read.est_num <= pos:
                    break
            if pos < 10:
                for id in a:
                    read.code_error(id, error=error)
            else:
                for id in c:
                    read.code_error(id, error=error)
    # read.export()
    results = analyze(read)
    print(results)
    return read

def Rand(type, stop='true', error='none', interval = 100000, seed=0):
    stopat = 1

    np.random.seed(seed)

    read = MAR()
    read = read.create("vuls_data_new.csv",type)

    read.interval = interval
    read.step = 10

    num2 = read.get_allpos()
    target = int(num2 * stopat)

    read.enable_est = True

    result = {}
    result['est'] = {'x':[],'semi':[]}
    while True:
        pos, neg, total = read.get_numbers()
        # try:
        #     print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        # except:
        #     print("%d, %d" %(pos,pos+neg))

        if pos + neg >= total or pos>=target:
            break


        for id in read.random():
            read.code_error(id, error=error)
        if pos+neg>0:
            result['est']['x'].append(pos+neg)
            result['est']['semi'].append(pos/(pos+neg)*total)

    result['pos'] = (read.record)

    return read

########
def BM25_est(filename, stop='true', error='none', interval = 100000, seed=0):
    stopat = 0.99
    thres = 0
    starting = 1
    counter = 0
    pos_last = 0
    np.random.seed(seed)

    read = MAR()
    read.crash='append'
    read = read.create(filename)

    read.interval = interval


    num2 = read.get_allpos()
    target = int(num2 * stopat)

    read.enable_est = True


    while True:
        pos, neg, total = read.get_numbers()
        try:
            print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        except:
            print("%d, %d" %(pos,pos+neg))

        if pos + neg >= total:
            if stop=='knee' and error=='random':
                coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                seq = coded[np.argsort(read.body['time'][coded])]
                part1 = set(seq[:read.kneepoint * read.step]) & set(
                    np.where(np.array(read.body['code']) == "no")[0])
                part2 = set(seq[read.kneepoint * read.step:]) & set(
                    np.where(np.array(read.body['code']) == "yes")[0])
                for id in part1 | part2:
                    read.code_error(id, error=error)
            break

        if pos < starting or pos+neg<thres:
            for id in read.BM25_get():
                read.code_error(id, error=error)
        else:
            a,b,c,d =read.train(weighting=True,pne=True)
            if stop == 'est':
                if stopat * read.est_num <= pos:
                    break
            elif stop == 'soft':
                if pos>0 and pos_last==pos:
                    counter = counter+1
                else:
                    counter=0
                pos_last=pos
                if counter >=5:
                    break
            elif stop == 'knee':
                if pos>0:
                    if read.knee():
                        if error=='random':
                            coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                            seq = coded[np.argsort(np.array(read.body['time'])[coded])]
                            part1 = set(seq[:read.kneepoint * read.step]) & set(
                                np.where(np.array(read.body['code']) == "no")[0])
                            part2 = set(seq[read.kneepoint * read.step:]) & set(
                                np.where(np.array(read.body['code']) == "yes")[0])
                            for id in part1|part2:
                                read.code_error(id, error=error)
                        break
            else:
                if pos >= target:
                    break
            if pos < 10:
                for id in a:
                    read.code_error(id, error=error)
            else:
                for id in c:
                    read.code_error(id, error=error)
    # read.export()
    # results = analyze(read)
    # print(results)
    result = {}
    result['est'] = (read.record_est)
    result['pos'] = (read.record)
    with open("../dump/"+filename+"_est.pickle","wb") as handle:
        pickle.dump(result,handle)
    return read

def Random_est(filename, stop='est', error='none', interval = 100000, seed=0):
    stopat = 0.99
    thres = 0
    starting = 1
    counter = 0
    pos_last = 0
    np.random.seed(seed)

    read = MAR()
    read = read.create(filename)

    read.interval = interval


    num2 = read.get_allpos()
    target = int(num2 * stopat)

    read.enable_est = True


    while True:
        pos, neg, total = read.get_numbers()
        try:
            print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        except:
            print("%d, %d" %(pos,pos+neg))

        if pos + neg >= total:
            if stop=='knee' and error=='random':
                coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                seq = coded[np.argsort(read.body['time'][coded])]
                part1 = set(seq[:read.kneepoint * read.step]) & set(
                    np.where(np.array(read.body['code']) == "no")[0])
                part2 = set(seq[read.kneepoint * read.step:]) & set(
                    np.where(np.array(read.body['code']) == "yes")[0])
                for id in part1 | part2:
                    read.code_error(id, error=error)
            break

        if pos < starting or pos+neg<thres:
            for id in read.random():
                read.code_error(id, error=error)
        else:
            a,b,c,d =read.train(weighting=True,pne=True)
            if stop == 'est':
                if stopat * read.est_num <= pos:
                    break
            elif stop == 'soft':
                if pos>0 and pos_last==pos:
                    counter = counter+1
                else:
                    counter=0
                pos_last=pos
                if counter >=5:
                    break
            elif stop == 'knee':
                if pos>0:
                    if read.knee():
                        if error=='random':
                            coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                            seq = coded[np.argsort(np.array(read.body['time'])[coded])]
                            part1 = set(seq[:read.kneepoint * read.step]) & set(
                                np.where(np.array(read.body['code']) == "no")[0])
                            part2 = set(seq[read.kneepoint * read.step:]) & set(
                                np.where(np.array(read.body['code']) == "yes")[0])
                            for id in part1|part2:
                                read.code_error(id, error=error)
                        break
            else:
                if pos >= target:
                    break
            if pos < 10:
                for id in a:
                    read.code_error(id, error=error)
            else:
                for id in c:
                    read.code_error(id, error=error)
    # read.export()
    # results = analyze(read)
    # print(results)
    result = {}
    result['est'] = (read.record_est)
    result['pos'] = (read.record)
    with open("../dump/"+filename+"_nocrash_est.pickle","wb") as handle:
        pickle.dump(result,handle)
    return read

########

def summaries():

    def get_stop(file,total,folder='dump'):

        with open("../"+folder+"/"+str(file)+".pickle","r") as handle:
            record_all = pickle.load(handle)
        with open("../"+folder+"/"+str(file)+"_nocrash.pickle","r") as handle:
            record_all2 = pickle.load(handle)
        with open("../"+folder+"/"+str(file)+"_crash.pickle","r") as handle:
            record_all3 = pickle.load(handle)
        with open("../"+folder+"/"+str(file)+"_metrics.pickle","r") as handle:
            record_all4 = pickle.load(handle)
        with open("../"+folder+"/"+str(file)+"_random.pickle","r") as handle:
            record_all5 = pickle.load(handle)

        result={}


        record = record_all['pos']

        x=np.array(record['x'])/int(total)
        y=np.array(record['pos'])/record['pos'][-1]

        record2 = record_all2['pos']

        x2=np.array(record2['x'])/int(total)
        y2=np.array(record2['pos'])/record['pos'][-1]

        record3 = record_all3['pos']

        x3=np.array(record3['x'])/int(total)
        y3=np.array(record3['pos'])/record['pos'][-1]

        record4 = record_all4['pos']

        x4=np.array(record4['x'])/int(total)
        y4=np.array(record4['pos'])/record['pos'][-1]

        record5 = record_all5['pos']

        x5=np.array(record5['x'])/int(total)
        y5=np.array(record5['pos'])/record['pos'][-1]

        re = {}
        for i in xrange(len(x)):
            if y[i]>=0.6 and not 0.6 in re:
                re[0.6]=x[i]
            if y[i]>=0.7 and not 0.7 in re:
                re[0.7]=x[i]
            if y[i]>=0.8 and not 0.8 in re:
                re[0.8]=x[i]
            if y[i]>=0.85 and not 0.85 in re:
                re[0.85]=x[i]
            if y[i]>=0.9 and not 0.9 in re:
                re[0.9]=x[i]
            if y[i]>=0.95 and not 0.95 in re:
                re[0.95]=x[i]
            if y[i]>=0.99 and not 0.99 in re:
                re[0.99]=x[i]
            if y[i]>=1 and not 1 in re:
                re[1]=x[i]


        result['combine']=re

        x=x2
        y=y2

        re = {}
        for i in xrange(len(x)):
            if y[i]>=0.6 and not 0.6 in re:
                re[0.6]=x[i]
            if y[i]>=0.7 and not 0.7 in re:
                re[0.7]=x[i]
            if y[i]>=0.8 and not 0.8 in re:
                re[0.8]=x[i]
            if y[i]>=0.85 and not 0.85 in re:
                re[0.85]=x[i]
            if y[i]>=0.9 and not 0.9 in re:
                re[0.9]=x[i]
            if y[i]>=0.95 and not 0.95 in re:
                re[0.95]=x[i]
            if y[i]>=0.99 and not 0.99 in re:
                re[0.99]=x[i]
            if y[i]>=1 and not 1 in re:
                re[1]=x[i]


        result['text']=re

        x=x3
        y=y3

        re = {}
        for i in xrange(len(x)):
            if y[i]>=0.6 and not 0.6 in re:
                re[0.6]=x[i]
            if y[i]>=0.7 and not 0.7 in re:
                re[0.7]=x[i]
            if y[i]>=0.8 and not 0.8 in re:
                re[0.8]=x[i]
            if y[i]>=0.85 and not 0.85 in re:
                re[0.85]=x[i]
            if y[i]>=0.9 and not 0.9 in re:
                re[0.9]=x[i]
            if y[i]>=0.95 and not 0.95 in re:
                re[0.95]=x[i]
            if y[i]>=0.99 and not 0.99 in re:
                re[0.99]=x[i]
            if y[i]>=1 and not 1 in re:
                re[1]=x[i]


        result['crash']=re

        x=x4
        y=y4

        re = {}
        for i in xrange(len(x)):
            if y[i]>=0.6 and not 0.6 in re:
                re[0.6]=x[i]
            if y[i]>=0.7 and not 0.7 in re:
                re[0.7]=x[i]
            if y[i]>=0.8 and not 0.8 in re:
                re[0.8]=x[i]
            if y[i]>=0.85 and not 0.85 in re:
                re[0.85]=x[i]
            if y[i]>=0.9 and not 0.9 in re:
                re[0.9]=x[i]
            if y[i]>=0.95 and not 0.95 in re:
                re[0.95]=x[i]
            if y[i]>=0.99 and not 0.99 in re:
                re[0.99]=x[i]
            if y[i]>=1 and not 1 in re:
                re[1]=x[i]


        result['metrics']=re

        x=x5
        y=y5
        re = {}
        for i in xrange(len(x)):
            if y[i]>=0.6 and not 0.6 in re:
                re[0.6]=x[i]
            if y[i]>=0.7 and not 0.7 in re:
                re[0.7]=x[i]
            if y[i]>=0.8 and not 0.8 in re:
                re[0.8]=x[i]
            if y[i]>=0.85 and not 0.85 in re:
                re[0.85]=x[i]
            if y[i]>=0.9 and not 0.9 in re:
                re[0.9]=x[i]
            if y[i]>=0.95 and not 0.95 in re:
                re[0.95]=x[i]
            if y[i]>=0.99 and not 0.99 in re:
                re[0.99]=x[i]
            if y[i]>=1 and not 1 in re:
                re[1]=x[i]


        result['random']=re


        return result

    files = {'dom':3505,'js':1421,'netwerk':698,'gfx':4814,'other':18312,'new':28750}
    record = {}
    for file in files:
        record[file] = get_stop('vuls_data_'+file+'.csv',files[file])

    keys = ['combine',  'text', 'metrics', 'crash', 'random']
    recalls = [0.6,0.7,0.8,0.85,0.9,0.95,0.99,1]
    modules = ['dom','netwerk','js','gfx','other','new']
    print("\\begin{tabular}{l|l|"+'c|'*len(record['dom']['text'])+"}")
    print("& Recall & "+ ' & '.join(map(str,recalls))+'\\\\\\hline')
    for key in keys:
        print('\\multirow{4}{*}{'+key+'}',end='')
        for m in modules:
            print(' & '+m,end='')
            for r in recalls:
                print(' & ',end='')
                try:
                    print(str(round(record[m][key][r],2)),end='')
                except:
                    print("N/A",end='')
            print('\\\\')
        print('\\hline')
    print('\\end{tabular}')


def sum_est():
    recalls = [0.9,0.95,0.99]
    files = {'dom':3505,'js':1421,'netwerk':698,'gfx':4814,'other':18312,'new':28750}
    modules = ['dom','netwerk','js','gfx','other','new']
    keys = ['combine',  'text', 'random']
    result = {}
    for file in files:
        result[file]={'combine':{},  'text':{}, 'random':{}}
        if file=='dom' or file=='new':
            for r in recalls:
                result[file]['combine'][r] = stop_at(r,'vuls_data_'+file+'.csv',files[file])
                result[file]['text'][r] = stop_at(r,'vuls_data_'+file+'.csv_nocrash',files[file])
                result[file]['random'][r] = stop_at(r,'vuls_data_'+file+'.csv_random',files[file])
        else:
            for r in recalls:
                result[file]['combine'][r] = stop_at(r,'vuls_data_'+file+'.csv_est',files[file])
                result[file]['text'][r] = stop_at(r,'vuls_data_'+file+'.csv_nocrash_est',files[file])
                result[file]['random'][r] = stop_at(r,'vuls_data_'+file+'.csv_random',files[file])

    print("\\begin{tabular}{l|l|"+'c|c|'*len(result['dom']['text'])+"}")
    print("& Desired & \\multicolumn{2}{c|}{"+ '} & \\multicolumn{2}{c|}{'.join(map(str,recalls))+'}\\\\\\cline{3-'+str(len(result['dom']['text'])*2+2)+'}')
    print("& Recall "+" & Recall & Effort "*len(result['dom']['text']) + '\\\\\\hline')
    for key in keys:
        print('\\multirow{4}{*}{'+key+'}',end='')
        for m in modules:
            print(' & '+m,end='')
            for r in recalls:
                try:
                    print(' & ',end='')
                    print(str(round(result[m][key][r]['recall'],2)),end='')
                    print(' & ',end='')
                    print(str(round(result[m][key][r]['effort'],2)),end='')
                except:
                    print(' & ',end='')
                    print(str(1.00),end='')
                    print(' & ',end='')
                    print(str(1.00),end='')
            print('\\\\')
        print('\\hline')
    print('\\end{tabular}')



def exp_sim():
    files = ['vuls_data_dom.csv','vuls_data_js.csv','vuls_data_netwerk.csv','vuls_data_gfx.csv','vuls_data_other.csv','vuls_data_new.csv']
    # files = ['vuls_data_other.csv']
    for file in files:
        # BM25(file)
        # Random(file)
        # CRASH(file)
        # Metrics(file)
        Rand(file)

def plot_act(file,total=28750):
    with open("../dump/"+str(file)+".pickle","r") as handle:
        record_all = pickle.load(handle)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 30}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 28, 'axes.labelsize': 40, 'legend.frameon': True,
             'figure.autolayout': False, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    plt.figure(1)

    # ax = plt.subplot(111)
    record = record_all['pos']
    record_est = record_all['est']

    x=np.array(record['x'])/int(total)
    y=np.array(record['pos'])/record['pos'][-1]


    re = {}
    pre = {}
    for i in xrange(len(x)):
        if y[i]>=0.6 and not 0.6 in re:
            re[0.6]=x[i]
            pre[0.6]=record['pos'][i]/record['x'][i]
        if y[i]>=0.7 and not 0.7 in re:
            re[0.7]=x[i]
            pre[0.7]=record['pos'][i]/record['x'][i]
        if y[i]>=0.8 and not 0.8 in re:
            re[0.8]=x[i]
            pre[0.8]=record['pos'][i]/record['x'][i]
        if y[i]>=0.85 and not 0.85 in re:
            re[0.85]=x[i]
            pre[0.85]=record['pos'][i]/record['x'][i]
        if y[i]>=0.9 and not 0.9 in re:
            re[0.9]=x[i]
            pre[0.9]=record['pos'][i]/record['x'][i]
        if y[i]>=0.95 and not 0.95 in re:
            re[0.95]=x[i]
            pre[0.95]=record['pos'][i]/record['x'][i]
        if y[i]>=1 and not 1 in re:
            re[1]=x[i]
            pre[1]=record['pos'][i]/record['x'][i]
    # print(re)
    # print(pre)
    # print('|---| Recall | '+ ' | '.join(map(str,sorted(pre.keys())))+' |')
    print("|"+file+'| Effort | '+ ' | '.join(map("{0:.3f}".format,sorted(re.values())))+"|")
    print("|"+file+'| Precision | '+ ' | '.join(map("{0:.3f}".format,sorted(pre.values(),reverse=True)))+"|")
    set_trace()
    plt.plot(x,y)

    plt.subplots_adjust(top=0.95, left=0.15, bottom=0.2, right=0.95)

    plt.ylabel("Recall")
    plt.xlabel("Effort")
    plt.savefig("../figure/" + str(file) + ".eps")
    plt.savefig("../figure/" + str(file) + ".png")

    plt.figure(2)
    x=np.array(record_est['x'])/int(total)
    y=np.array(record_est['semi'])/record['pos'][-1]
    plt.plot(x,y)

    plt.plot(x,[1]*len(x),linestyle = '--')

    plt.subplots_adjust(top=0.95, left=0.15, bottom=0.2, right=0.95)

    plt.ylabel("Estimation")
    plt.xlabel("Effort")
    plt.savefig("../figure/est_" + str(file) + ".eps")
    plt.savefig("../figure/est_" + str(file) + ".png")

def plot_est(file,total=28750):
    with open("../dump/"+str(file)+"_est.pickle","r") as handle:
        record_all = pickle.load(handle)
    with open("../dump/"+str(file)+"_nocrash_est.pickle","r") as handle:
        record_all2 = pickle.load(handle)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 30}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 28, 'axes.labelsize': 40, 'legend.frameon': True,
             'figure.autolayout': False, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    plt.figure(1)

    # ax = plt.subplot(111)
    record = record_all['pos']
    record_est = record_all['est']




    record_est2 = record_all2['est']




    plt.figure(2)
    x=np.array(record_est['x'])/int(total)
    y=np.array(record_est['semi'])/record['pos'][-1]
    x2=np.array(record_est2['x'])/int(total)
    y2=np.array(record_est2['semi'])/record['pos'][-1]

    plt.plot(x,y,label='combine')
    plt.plot(x2,y2,linestyle = '-.',label='text')

    plt.plot(x,[1]*len(x),linestyle = '--',label = 'true')

    plt.subplots_adjust(top=0.95, left=0.15, bottom=0.2, right=0.95)
    plt.legend(bbox_to_anchor=(0.9, 0.80), loc=1, ncol=1, borderaxespad=0.)

    plt.ylabel("Estimation")
    plt.xlabel("Effort")
    plt.savefig("../figure/est_" + str(file) + ".eps")
    plt.savefig("../figure/est_" + str(file) + ".png")
    plt.close()

def plot_compare(file,total=28750):
    with open("../dump/"+str(file)+".pickle","r") as handle:
        record_all = pickle.load(handle)
    with open("../dump/"+str(file)+"_nocrash.pickle","r") as handle:
        record_all2 = pickle.load(handle)
    with open("../dump/"+str(file)+"_crash.pickle","r") as handle:
        record_all3 = pickle.load(handle)
    with open("../dump/"+str(file)+"_metrics.pickle","r") as handle:
        record_all4 = pickle.load(handle)
    with open("../dump/"+str(file)+"_random.pickle","r") as handle:
        record_all5 = pickle.load(handle)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 30}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 30, 'axes.labelsize': 40, 'legend.frameon': True,
             'figure.autolayout': False, 'figure.figsize': (16, 10)}
    plt.rcParams.update(paras)

    plt.figure(1)
    ax = plt.subplot(111)

    record = record_all['pos']
    record_est = record_all['est']

    x=np.array(record['x'])/int(total)
    y=np.array(record['pos'])/record['pos'][-1]

    record2 = record_all2['pos']
    record_est2 = record_all2['est']

    x2=np.array(record2['x'])/int(total)
    y2=np.array(record2['pos'])/record['pos'][-1]

    record3 = record_all3['pos']
    record_est3 = record_all3['est']

    x3=np.array(record3['x'])/int(total)
    y3=np.array(record3['pos'])/record['pos'][-1]

    record4 = record_all4['pos']
    record_est4 = record_all4['est']

    x4=np.array(record4['x'])/int(total)
    y4=np.array(record4['pos'])/record['pos'][-1]

    record5 = record_all5['pos']
    record_est5 = record_all5['est']

    x5=np.array(record5['x'])/int(total)
    y5=np.array(record5['pos'])/record['pos'][-1]

    # re = {}
    # pre = {}
    # for i in xrange(len(x)):
    #     if y[i]>=0.6 and not 0.6 in re:
    #         re[0.6]=x[i]
    #         pre[0.6]=record['pos'][i]/record['x'][i]
    #     if y[i]>=0.7 and not 0.7 in re:
    #         re[0.7]=x[i]
    #         pre[0.7]=record['pos'][i]/record['x'][i]
    #     if y[i]>=0.8 and not 0.8 in re:
    #         re[0.8]=x[i]
    #         pre[0.8]=record['pos'][i]/record['x'][i]
    #     if y[i]>=0.85 and not 0.85 in re:
    #         re[0.85]=x[i]
    #         pre[0.85]=record['pos'][i]/record['x'][i]
    #     if y[i]>=0.9 and not 0.9 in re:
    #         re[0.9]=x[i]
    #         pre[0.9]=record['pos'][i]/record['x'][i]
    #     if y[i]>=0.95 and not 0.95 in re:
    #         re[0.95]=x[i]
    #         pre[0.95]=record['pos'][i]/record['x'][i]
    #     if y[i]>=1 and not 1 in re:
    #         re[1]=x[i]
    #         pre[1]=record['pos'][i]/record['x'][i]
    # # print(re)
    # # print(pre)
    # # print('|---| Recall | '+ ' | '.join(map(str,sorted(pre.keys())))+' |')
    # print("|"+file+'| Effort | '+ ' | '.join(map("{0:.3f}".format,sorted(re.values())))+"|")
    # print("|"+file+'| Precision | '+ ' | '.join(map("{0:.3f}".format,sorted(pre.values(),reverse=True)))+"|")

    ax.plot(x,y,label='Combine')
    ax.plot(x2,y2,linestyle = '-.',label='Text')
    ax.plot(x4,y4,linestyle = ':', color = 'black', label='Metrics')
    ax.plot(x3,y3,linestyle = '--',label='Theisen\'15')
    ax.plot(x5,y5,dashes=[10, 5, 20, 5],label='Random')

    # plt.subplots_adjust(top=0.95, left=0.15, bottom=0.2, right=0.95)
    # plt.legend(bbox_to_anchor=(0.9, 0.70), loc=1, ncol=1, borderaxespad=0.)
    plt.subplots_adjust(top=0.95, left=0.10, bottom=0.15, right=0.72)

    ax.legend(bbox_to_anchor=(1.02, 1), loc=2, ncol=1, borderaxespad=0.)

    plt.ylabel("Recall")
    plt.xlabel("Cost")
    plt.savefig("../figure/" + str(file.split('.')[0]) + ".eps")
    plt.savefig("../figure/" + str(file.split('.')[0]) + ".png")

    plt.close()

    plt.figure(2)
    ax = plt.subplot(111)
    x=np.array(record_est['x'])/int(total)
    y=np.array(record_est['semi'])/record['pos'][-1]
    x2=np.array(record_est2['x'])/int(total)
    y2=np.array(record_est2['semi'])/record['pos'][-1]
    x3=np.array(record_est3['x'])/int(total)
    y3=np.array(record_est3['semi'])/record['pos'][-1]
    x4=np.array(record_est4['x'])/int(total)
    y4=np.array(record_est4['semi'])/record['pos'][-1]
    x5=np.array(record_est5['x'])/int(total)
    y5=np.array(record_est5['semi'])/record['pos'][-1]

    ax.plot(x,y,label='Combine')
    ax.plot(x2[:len(x)],y2[:len(x)],linestyle = '-.',label='Text')
    ax.plot(x5[10:len(x)*10],y5[10:len(x)*10],linestyle = ':',label='Random')
    # plt.plot(x3,y3,linestyle = ':',label='crash')

    ax.plot(x,[1]*len(x),linestyle = '--',label = 'True')

    # plt.subplots_adjust(top=0.95, left=0.15, bottom=0.2, right=0.95)
    # plt.legend(bbox_to_anchor=(0.9, 1), loc=1, ncol=1, borderaxespad=0.)

    plt.subplots_adjust(top=0.95, left=0.10, bottom=0.15, right=0.72)

    ax.legend(bbox_to_anchor=(1.02, 1), loc=2, ncol=1, borderaxespad=0.)

    plt.ylabel("Estimation")
    plt.xlabel("Cost")
    plt.savefig("../figure/est_" + str(file.split('.')[0]) + ".eps")
    plt.savefig("../figure/est_" + str(file.split('.')[0]) + ".png")
    plt.close()

def stop_at(target, file,total=28750):
    with open("../dump/"+str(file)+".pickle","r") as handle:
        record_all = pickle.load(handle)
    t= float(target)
    comp=0
    for i in xrange(len(record_all['pos']['x'])-1):
        if record_all['pos']['x'][i]!=record_all['est']['x'][i-comp]:
            comp+=1
            continue
        if record_all['pos']['pos'][i]==0:
            continue
        stop = int(t*record_all['est']['semi'][i-comp])<=record_all['pos']['pos'][i]

        if stop:

            print(record_all['pos']['x'][i])
            print(record_all['pos']['pos'][i])
            return {'effort': record_all['pos']['x'][i]/total, 'recall': record_all['pos']['pos'][i]/record_all['pos']['pos'][-1]}

def run_est():
    files = ['vuls_data_js.csv','vuls_data_netwerk.csv','vuls_data_gfx.csv','vuls_data_other.csv']
    # files = ['vuls_data_other.csv']
    for file in files:
        BM25_est(file,'est')
        Random_est(file,'est')

def lda_plot(file):
    read = MAR()
    read = read.create(file)
    read.lda()
    set_trace()
    read.plot_lda()
    set_trace()

def auto_plot():
    files = {'dom':3505,'js':1421,'netwerk':698,'gfx':4814,'other':18312,'new':28750}
    for file in files:
        plot_compare('vuls_data_'+file+'.csv',files[file])

def auto_plot2():
    files = {'dom':3505,'js':1421,'netwerk':698,'gfx':4814,'other':18312,'new':28750}
    for file in files:
        if file == 'dom' or file == 'new':
            continue
        else:
            plot_est('vuls_data_'+file+'.csv',files[file])


## HPC




def error_hpcc_feature(fea, seed = 1):
    seed = int(seed)
    np.random.seed(seed)
    types = ['Arbitrary Code', 'Improper Control of a Resource Through its Lifetime', 'Other', 'Range Error', 'Code Quality', 'all']
    # types = ['all']


    results={}

    for type in types:
        try:
            with open("../dump/features_"+str(fea)+"_hpcc_"+str(seed)+".pickle","r") as handle:
                results = pickle.load(handle)
        except:
            pass
        print(str(seed)+": "+type+": "+ fea+ ": ", end='')
        if fea == 'combine':
            result = BM25(type,stop='mix',seed=seed)
        elif fea == 'text':
            result = Random(type,stop='mix',seed=seed)
        elif fea == 'metrics':
            result = Metrics(type,stop='true',seed=seed)
        elif fea == 'random':
            result = Rand(type,stop='true',seed=seed)
        elif fea == 'crash':
            result = CRASH(type,stop='true',seed=seed)
        else:
            result = Rand(type,stop='true',seed=seed)

        results[type] = {'pos':result.record,'est':result.record_est}
        with open("../dump/features_"+str(fea)+"_hpcc_"+str(seed)+".pickle","w") as handle:
            pickle.dump(results,handle)

def error_hpcc_more(cor, error_rate = 0.5, seed = 1):
    error_rate = float(error_rate)
    seed = int(seed)
    np.random.seed(seed)
    files = ['vuls_data_dom.csv','vuls_data_js.csv','vuls_data_netwerk.csv','vuls_data_gfx.csv','vuls_data_other.csv','vuls_data_new.csv']
    # files = ['vuls_data_new.csv']
    # files = ['vuls_data_dom.csv','vuls_data_js.csv','vuls_data_netwerk.csv','vuls_data_gfx.csv','vuls_data_other.csv']

    # correct = ['machine']

    Runfunc = Random

    results={}

    for file in files:
        print(str(seed)+": "+file+": "+ cor+ ": ", end='')
        if cor == 'majority':
            result = Runfunc(file,stop='est',error='three',error_rate = error_rate, seed=seed)
        elif cor == 'machine':
            result = Runfunc(file,stop='est',error='random',error_rate = error_rate, correct = 'machine', interval = 1, seed=seed)
        elif cor == 'knee':
            result = Runfunc(file,stop='knee',error='random',error_rate = error_rate, correct = 'no', seed=seed)
        elif cor == 'random':
            result = Runfunc(file,stop='est',error='random',error_rate = error_rate, correct = 'random', interval = 1, seed=seed)
        elif cor == 'machine3':
            result = Runfunc(file,stop='est',error='random3',error_rate = error_rate, correct = 'machine', interval = 1, seed=seed)
        else:
            result = Runfunc(file,stop='est',error='random', error_rate = error_rate, seed=seed)

        results[file] = analyze(result)
        with open("../dump/error_"+str(cor)+"_hpcc"+str(int(error_rate*100))+"_"+str(seed)+".pickle","w") as handle:
            pickle.dump(results,handle)

def error_hpcc(error_rate = 0.5, seed = 1):
    error_rate = float(error_rate)
    seed = int(seed)
    np.random.seed(seed)
    files = ['vuls_data_dom.csv','vuls_data_js.csv','vuls_data_netwerk.csv','vuls_data_gfx.csv','vuls_data_other.csv','vuls_data_new.csv']
    # files = ['vuls_data_new.csv']
    # files = ['vuls_data_dom.csv','vuls_data_js.csv','vuls_data_netwerk.csv','vuls_data_gfx.csv','vuls_data_other.csv']
    correct = ['majority', 'machine', 'knee','none']
    # correct = ['machine']
    crash = 'new'
    if crash=='crash':
        Runfunc = BM25
    else:
        Runfunc = Random

    results={file:{} for file in files}
    for cor in correct:

        for file in files:
            print(str(seed)+": "+file+": "+ cor+ ": ", end='')
            if cor == 'majority':
                result = Runfunc(file,stop='est',error='three',error_rate = error_rate, seed=seed)
            elif cor == 'machine':
                result = Runfunc(file,stop='est',error='random',error_rate = error_rate, correct = 'machine', interval = 1, seed=seed)
            elif cor == 'knee':
                result = Runfunc(file,stop='knee',error='random',error_rate = error_rate, correct = 'no', seed=seed)
            elif cor == 'random':
                result = Runfunc(file,stop='est',error='random',error_rate = error_rate, correct = 'random', interval = 1, seed=seed)
            else:
                result = Runfunc(file,stop='est',error='random', error_rate = error_rate, seed=seed)

            results[file][cor] = analyze(result)
            with open("../dump/error_"+str(crash)+"_hpcc"+str(int(error_rate*100))+"_"+str(seed)+".pickle","w") as handle:
                pickle.dump(results,handle)


def error_inc(neg_len = 0.4, crash = 'new'):

    files = ['vuls_data_dom.csv','vuls_data_js.csv','vuls_data_netwerk.csv','vuls_data_gfx.csv','vuls_data_other.csv','vuls_data_new.csv']
    correct = ['machine']

    if crash=='crash':
        Runfunc = BM25
    else:
        Runfunc = Random

    results={}
    tmp = []
    for file in files:
        results[file]={}
        for cor in correct:
            results[file][cor]={}
            print(str(neg_len)+": "+file+": "+ cor+ ": ", end='')
            for i in xrange(10):
                result = Runfunc(file,stop='est',error='three', seed=i, neg_len=float(neg_len))
                tmp.append(analyze(result))
            for key in analyze(result):
                results[file][cor][key] = np.median([t[key] for t in tmp])
    with open("../dump/inc_"+str(crash)+"_hpcc30_"+str(neg_len)+".pickle","w") as handle:
        pickle.dump(results,handle)

def error_summary():
    # import cPickle as pickle

    crashes = ['new']
    errors = ['00','10','20','30','40','50']
    # errors = ['0','10','20','30']
    for crash in crashes:
        for error in errors:
            results = []
            for i in xrange(60):
                try:
                    with open("../dump/error_"+str(crash)+"_hpcc"+str(error)+"_"+str(i)+".pickle","r") as handle:
                        results.append(pickle.load(handle))
                except:
                    pass
                # results=[]
                # for i in xrange(30):
                #     results.append(pickle.load(handle))

            with open("../dump/error2_"+str(crash)+"_hpcc"+str(error)+'.pickle', 'w') as f:
                pickle.dump(results,f)

def error_summary_new():
    # import cPickle as pickle

    methods = ['none','majority','machine','knee','machine3','random']
    errors = ['0','10','20','30','40','50']
    # errors = ['0','10','20','30']
    files = ['vuls_data_dom.csv','vuls_data_js.csv','vuls_data_netwerk.csv','vuls_data_gfx.csv','vuls_data_other.csv','vuls_data_new.csv']

    for error in errors:
        results = {file:{m:{} for m in methods} for file in files}
        for method in methods:

            for i in xrange(30):
                try:
                    with open("../dump/error_"+str(method)+"_hpcc"+str(error)+"_"+str(i)+".pickle","r") as handle:
                        tmp = (pickle.load(handle))
                except:
                    continue

                for f in tmp:
                    for key in tmp[f].keys():

                        if not key in results[f][method]:
                            results[f][method][key] = []
                        results[f][method][key].append(tmp[f][key])




        with open("../dump/error2_new_hpcc"+str(error)+'.pickle', 'w') as f:
                pickle.dump(results,f)


def error_comp():
    files = ['vuls_data_dom.csv', 'vuls_data_js.csv', 'vuls_data_netwerk.csv', 'vuls_data_gfx.csv',
             'vuls_data_other.csv', 'vuls_data_new.csv']
    name = {'vuls_data_dom.csv': 'Module: dom', 'vuls_data_js.csv': 'Module: js', 'vuls_data_netwerk.csv': 'Module: netwerk', 'vuls_data_gfx.csv': 'Module: gfx',
             'vuls_data_other.csv': 'Other modules', 'vuls_data_new.csv': 'Entire project', 'Median': 'Median'}
    correct = ['none', 'majority', 'machine', 'knee']
    total = {'vuls_data_dom.csv': 86, 'vuls_data_js.csv': 57, 'vuls_data_netwerk.csv': 29, 'vuls_data_gfx.csv': 28,
             'vuls_data_other.csv': 71, 'vuls_data_new.csv': 271}
    total2 = {'vuls_data_dom.csv': 3505,'vuls_data_js.csv': 1421,'vuls_data_netwerk.csv': 698,'vuls_data_gfx.csv': 4814,'vuls_data_other.csv': 18312,'vuls_data_new.csv':28750}
    metrics = ['Recall', 'Cost']

    correct2 = ['majority','Disagree (3)', 'Two-person (0.5)', 'machine']



    ## Baseline
    with open("../dump/error2_new_hpcc00.pickle", 'r') as f:
        results = pickle.load(f)


    cols = ["Dataset"]+metrics
    bl = {col:[] for col in cols}
    bl0 = {}


    trans = {'Median':{}}

    for file in files:
        trans[file] = {}
        for cor in correct:
            trans[file][cor] = {}
            keys = results[0][file][cor].keys()
            for key in keys:
                trans[file][cor][key]=[]
                for i in xrange(len(results)):
                    trans[file][cor][key].append(results[i][file][cor][key])

    medians={'recall':[],'cost':[]}
    for file in files:
        bl['Dataset'].append(name[file])
        bl0[file]={}
        # recall
        x = np.array(trans[file]['none']['truepos'])/total[file]
        medians['recall'].extend(x)
        median = str(int(np.median(x)*100))
        iqr = str(int((np.percentile(x,75)-np.percentile(x,25))*100))
        bl['Recall'].append(median+'('+iqr+')')
        bl0[file]['Recall'] = np.median(x)

        # precision
        # x = np.array(trans[file]['none']['truepos']) / (np.array(trans[file]['none']['truepos'])+np.array(trans[file]['none']['falsepos']))
        # median = str(int(np.median(x) * 100))
        # iqr = str(int((np.percentile(x, 75) - np.percentile(x, 25)) * 100))
        # bl['Precision'].append(median + '(' + iqr + ')')
        # bl0[file]['Precision'] = np.median(x)

        # cost
        x = np.array(trans[file]['none']['count']) / total2[file]
        median = str(int(np.median(x)*100))
        medians['cost'].extend(x)
        iqr = str(int((np.percentile(x,75)-np.percentile(x,25))*100))
        bl['Cost'].append(median + '(' + iqr + ')')
        bl0[file]['Cost'] = np.median(x)


    bl0['Median']={}

    bl['Dataset'].append('Median')
    median = str(int(np.median(medians['recall'])*100))
    iqr = str(int((np.percentile(medians['recall'],75)-np.percentile(medians['recall'],25))*100))
    bl['Recall'].append(median+'('+iqr+')')
    bl0['Median']['Recall'] = np.median(medians['recall'])

    median = str(int(np.median(medians['cost'])*100))
    iqr = str(int((np.percentile(medians['cost'],75)-np.percentile(medians['cost'],25))*100))
    bl['Cost'].append(median+'('+iqr+')')
    bl0['Median']['Cost'] = np.median(medians['cost'])

    # df = pd.DataFrame(data=bl, columns=cols)
    # df.to_csv("../dump/error_baseline.csv")

    ## all results

    cols = ["Dataset"]
    for m in metrics:
        for c in correct2:
            cols.append(m+'_'+c)



    errors = ['00','10','20','30','40','50']
    for error in errors:
        with open("../dump/error2_new_hpcc"+error+".pickle", 'r') as f:
            results = pickle.load(f)

        trans = {'Median':{}}

        for file in files:
            trans[file] = {}
            for cor in correct:
                trans[file][cor] = {}
                keys = results[0][file][cor].keys()
                for key in keys:
                    trans[file][cor][key]=[]
                    for i in xrange(len(results)):
                        trans[file][cor][key].append(results[i][file][cor][key])

        dictdf = {col: [] for col in cols}

        medians={}
        for file in files:
            dictdf['Dataset'].append(name[file])
            for cor in correct2:

                if not 'recall_'+cor in medians:
                    medians['recall_'+cor] = []
                if not 'cost_'+cor in medians:
                    medians['cost_'+cor] = []

                if cor == 'Two-person (0.5)':
                    # recall
                    x = (np.array(trans[file]['majority']['truepos'])+np.array(trans[file]['none']['truepos'])) / 2 / total[file] / bl0[file]['Recall']
                    medians['recall_'+cor].extend(x)
                    median = str(int(np.median(x)*100))
                    iqr = str(int((np.percentile(x,75)-np.percentile(x,25))*100))
                    dictdf['Recall_'+cor].append(median + '(' + iqr + ')')

                    # cost
                    x = (np.array(trans[file]['majority']['count'])+np.array(trans[file]['none']['count'])) / 2 / total2[file] / bl0[file]['Cost']
                    medians['cost_'+cor].extend(x)
                    median = str(int(np.median(x)*100))
                    iqr = str(int((np.percentile(x,75)-np.percentile(x,25))*100))
                    dictdf['Cost_'+cor].append(median + '(' + iqr + ')')
                elif cor == 'Disagree (3)':
                    # recall
                    err = int(error)/100.0
                    x = ((np.array(trans[file]['machine']['truepos'])-np.array(trans[file]['none']['truepos']))*(1-err**2)/(1-err)+np.array(trans[file]['none']['truepos']))  / total[file] / bl0[file]['Recall']
                    medians['recall_'+cor].extend(x)
                    median = str(int(np.median(x)*100))
                    iqr = str(int((np.percentile(x,75)-np.percentile(x,25))*100))
                    dictdf['Recall_'+cor].append(median + '(' + iqr + ')')

                    # cost
                    x = (2*np.array(trans[file]['machine']['count'])-np.array(trans[file]['none']['count']))  / total2[file] / bl0[file]['Cost']
                    medians['cost_'+cor].extend(x)
                    median = str(int(np.median(x)*100))
                    iqr = str(int((np.percentile(x,75)-np.percentile(x,25))*100))
                    dictdf['Cost_'+cor].append(median + '(' + iqr + ')')
                else:

                    # recall
                    x = np.array(trans[file][cor]['truepos']) / total[file] / bl0[file]['Recall']
                    medians['recall_'+cor].extend(x)
                    median = str(int(np.median(x)*100))
                    iqr = str(int((np.percentile(x,75)-np.percentile(x,25))*100))
                    dictdf['Recall_'+cor].append(median + '(' + iqr + ')')

                    # cost
                    x = np.array(trans[file][cor]['count']) / total2[file] / bl0[file]['Cost']
                    medians['cost_'+cor].extend(x)
                    median = str(int(np.median(x)*100))
                    iqr = str(int((np.percentile(x,75)-np.percentile(x,25))*100))
                    dictdf['Cost_'+cor].append(median + '(' + iqr + ')')


        dictdf['Dataset'].append('Median')
        for cor in correct2:
            median = str(int(np.median(medians['recall_'+cor])*100))
            iqr = str(int((np.percentile(medians['recall_'+cor],75)-np.percentile(medians['recall_'+cor],25))*100))
            dictdf['Recall_'+cor].append(median+'('+iqr+')')

            median = str(int(np.median(medians['cost_'+cor])*100))
            iqr = str(int((np.percentile(medians['cost_'+cor],75)-np.percentile(medians['cost_'+cor],25))*100))
            dictdf['Cost_'+cor].append(median+'('+iqr+')')


        # for file in files:
        #     dictdf['Dataset'].append(name[file])
        #     for cor in correct:
        #         # recall
        #         x = np.array(trans[file][cor]['truepos']) / total[file] / bl0[file]['Recall']
        #         median = str(int(np.median(x)*100))
        #         iqr = str(int((np.percentile(x,75)-np.percentile(x,25))*100))
        #         dictdf['Recall_'+cor].append(median + '(' + iqr + ')')
        #
        #
        #         # precision
        #         # x = np.array(trans[file][cor]['truepos']) / (
        #         # np.array(trans[file][cor]['truepos']) + np.array(trans[file][cor]['falsepos'])) / bl0[file]['Precision']
        #         # median = str(int(np.median(x)*100))
        #         # iqr = str(int((np.percentile(x,75)-np.percentile(x,25))*100))
        #         # dictdf['Precision_'+cor].append(median + '(' + iqr + ')')
        #
        #
        #         # cost
        #         x = np.array(trans[file][cor]['count']) / total2[file] / bl0[file]['Cost']
        #         median = str(int(np.median(x)*100))
        #         iqr = str(int((np.percentile(x,75)-np.percentile(x,25))*100))
        #         dictdf['Cost_'+cor].append(median + '(' + iqr + ')')

        df = pd.DataFrame(data=dictdf,
                          columns=cols)
        df.to_csv("../dump/error2_" + str(error) + '.csv')


def error_sum(crash="new",error='20'):
    files = ['vuls_data_dom.csv','vuls_data_js.csv','vuls_data_netwerk.csv','vuls_data_gfx.csv','vuls_data_other.csv','vuls_data_new.csv']
    correct = ['none', 'majority', 'machine', 'knee']
    total = {'vuls_data_dom.csv': 86,'vuls_data_js.csv': 57,'vuls_data_netwerk.csv': 29,'vuls_data_gfx.csv': 28,'vuls_data_other.csv': 71,'vuls_data_new.csv':271}
    results = []

    if error == 0:
        error = "00"


    with open("../dump/error2_"+str(crash)+"_hpcc"+str(error)+'.pickle', 'r') as f:
        results = pickle.load(f)


    trans = {}

    for file in files:
        trans[file] = {}
        for cor in correct:
            trans[file][cor] = {}
            keys = results[0][file][cor].keys()

            for key in keys:
                trans[file][cor][key]=[]
                for i in xrange(len(results)):
                    trans[file][cor][key].append(results[i][file][cor][key])

    ####draw table

    print("\\begin{tabular}{ |l|"+"c|"*len(correct)+" }")
    print("\\hline")
    print("  & "+" & ".join(correct)+"  \\\\")
    print("\\hline")
    for dataset in files:
        # out = dataset.split('.')[0]+" & " + ' & '.join([str(int(np.median(trans[dataset][cor]['truepos'])))+" / "+ str(int(np.median(trans[dataset][cor]['count']))) +" / "+ str(int(np.median(trans[dataset][cor]['falseneg']))) +" / "+ str(int(np.median(trans[dataset][cor]['falsepos']))) for cor in correct]) + '\\\\'
        out = dataset.split('.')[0]+" & " + ' & '.join([str(round(np.median(np.array(trans[dataset][cor]['truepos'])/total[dataset]),2))+" / "+str(round(np.median(np.array(trans[dataset][cor]['truepos'])/(np.array(trans[dataset][cor]['truepos'])+np.array(trans[dataset][cor]['falsepos']))),2))+" / "+str(int(np.median(trans[dataset][cor]['count']))) for cor in correct]) + '\\\\'
        print(out)
        print("\\hline")
    print("\\end{tabular}")

def error_sum2(error='20',crash="new"):
    files = ['vuls_data_dom.csv','vuls_data_js.csv','vuls_data_netwerk.csv','vuls_data_gfx.csv','vuls_data_other.csv','vuls_data_new.csv']
    correct = ['none', 'majority', 'machine', 'knee', 'active-knee']
    total = {'vuls_data_dom.csv': 86,'vuls_data_js.csv': 57,'vuls_data_netwerk.csv': 29,'vuls_data_gfx.csv': 28,'vuls_data_other.csv': 71,'vuls_data_new.csv':271}
    results = []

    if error == 0:
        error = "00"


    with open("../dump/error_"+str(crash)+"_hpcc"+str(error)+'.pickle', 'r') as f:
        results = pickle.load(f)


    trans = {}

    for file in files:
        trans[file] = {}
        for cor in correct:
            trans[file][cor] = {}
            keys = results[0][file][cor].keys()
            for key in keys:
                trans[file][cor][key]=[]
                for i in xrange(len(results)):
                    trans[file][cor][key].append(results[i][file][cor][key])

    dictdf = {"Dataset":[], "none_recall":[], "none_cost":[], "majority_recall":[], "majority_cost":[], "machine_recall":[], "machine_cost":[], "knee_recall":[], "knee_cost":[], "active-knee_recall":[], "active-knee_cost":[]}

    for dataset in files:
        dictdf['Dataset'].append(dataset)
        for cor in correct:
            dictdf[cor+"_recall"].append(str(round(np.median(np.array(trans[dataset][cor]['truepos'])/total[dataset]),2)))
            dictdf[cor+"_cost"].append(str(int(np.median(trans[dataset][cor]['count']))))
    df = pd.DataFrame(data=dictdf, columns=["Dataset", "none_recall", "none_cost", "majority_recall", "majority_cost", "machine_recall", "machine_cost", "knee_recall", "knee_cost", "active-knee_recall", "active-knee_cost"])
    df.to_csv("../dump/error_"+str(error)+'.csv')


    ####draw table

    print("\\begin{tabular}{ |l|"+"c|"*len(correct)+" }")
    print("\\hline")
    print("  & "+" & ".join(correct)+"  \\\\")
    print("\\hline")
    for dataset in files:
        # out = dataset.split('.')[0]+" & " + ' & '.join([str(int(np.median(trans[dataset][cor]['truepos'])))+" / "+ str(int(np.median(trans[dataset][cor]['count']))) +" / "+ str(int(np.median(trans[dataset][cor]['falseneg']))) +" / "+ str(int(np.median(trans[dataset][cor]['falsepos']))) for cor in correct]) + '\\\\'
        out = dataset.split('.')[0]+" & " + ' & '.join([str(round(np.median(np.array(trans[dataset][cor]['truepos'])/total[dataset]),2))+" / "+str(int(np.median(trans[dataset][cor]['count']))) for cor in correct]) + '\\\\'
        print(out)
        print("\\hline")
    print("\\end{tabular}")

def recall_cost(error='30',crash = 'new'):
    files = ['vuls_data_dom.csv','vuls_data_js.csv','vuls_data_netwerk.csv','vuls_data_gfx.csv','vuls_data_other.csv','vuls_data_new.csv']

    total = {'vuls_data_dom.csv': 86,'vuls_data_js.csv': 57,'vuls_data_netwerk.csv': 29,'vuls_data_gfx.csv': 28,'vuls_data_other.csv': 71,'vuls_data_new.csv':271}
    total2 = {'vuls_data_dom.csv': 1200,'vuls_data_js.csv': 600,'vuls_data_netwerk.csv': 500,'vuls_data_gfx.csv': 2400,'vuls_data_other.csv': 5100,'vuls_data_new.csv':6300}
    if error == 0:
        error = "00"


    font = {'family': 'normal',
            'weight': 'bold',
            'size': 30}
    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 30, 'axes.labelsize': 40, 'legend.frameon': True,
             'figure.autolayout': False, 'figure.figsize': (16, 10)}
    plt.rcParams.update(paras)



    results={}
    cor_rate = np.array(range(0,105,5))*0.01
    for i in cor_rate:

        with open("../dump/inc_"+str(crash)+"_hpcc"+str(error)+"_"+str(i)+'.pickle', 'r') as f:
            results[i] = pickle.load(f)

    rec = {}
    for file in files:
        plt.figure(1)
        ax = plt.subplot(111)
        rec[file]={'recall':[],'cost':[]}
        for i in cor_rate:
            rec[file]['recall'].append(results[i][file]['machine']['truepos']/total[file])
            rec[file]['cost'].append(results[i][file]['machine']['count']/total2[file])

        ax.plot(cor_rate,rec[file]['recall'],label='Recall')
        ax.plot(cor_rate,rec[file]['cost'],linestyle = '-.',label='Cost')



        plt.subplots_adjust(top=0.95, left=0.10, bottom=0.15, right=0.72)

        ax.legend(bbox_to_anchor=(1.02, 1), loc=2, ncol=1, borderaxespad=0.)

        # plt.ylabel("Estimation")
        plt.xlabel("Correction Rate")
        plt.savefig("../figure/recall_cost_" + str(file.split('.')[0]) + ".eps")
        plt.savefig("../figure/recall_cost_" + str(file.split('.')[0]) + ".png")
        plt.close()


def error_sk():
    files = ['vuls_data_dom.csv', 'vuls_data_js.csv', 'vuls_data_netwerk.csv', 'vuls_data_gfx.csv',
             'vuls_data_other.csv', 'vuls_data_new.csv']
    name = {'vuls_data_dom.csv': 'Module: dom', 'vuls_data_js.csv': 'Module: js', 'vuls_data_netwerk.csv': 'Module: netwerk', 'vuls_data_gfx.csv': 'Module: gfx',
             'vuls_data_other.csv': 'Other modules', 'vuls_data_new.csv': 'Entire project', 'Median': 'Median'}
    correct = ['none', 'majority', 'machine', 'knee']
    total = {'vuls_data_dom.csv': 86, 'vuls_data_js.csv': 57, 'vuls_data_netwerk.csv': 29, 'vuls_data_gfx.csv': 28,
             'vuls_data_other.csv': 71, 'vuls_data_new.csv': 271}
    total2 = {'vuls_data_dom.csv': 3505,'vuls_data_js.csv': 1421,'vuls_data_netwerk.csv': 698,'vuls_data_gfx.csv': 4814,'vuls_data_other.csv': 18312,'vuls_data_new.csv':28750}
    metrics = ['Recall', 'Cost']

    errors = ['00','10','20','30','40','50']
    for error in errors:
        print(str(error))
        with open("../dump/error2_new_hpcc"+error+".pickle", 'r') as f:
            results = pickle.load(f)

        trans = {'Median':{}}

        for file in files:
            trans[file] = {}
            for cor in correct:
                trans[file][cor] = {}
                keys = results[0][file][cor].keys()
                for key in keys:
                    trans[file][cor][key]=[]
                    for i in xrange(len(results)):
                        trans[file][cor][key].append(results[i][file][cor][key])

        cols = ["Dataset"]
        for m in metrics:
            for c in correct:
                cols.append(m+'_'+c)

        dictdf = {col: [] for col in cols}



        medians={}
        for file in files:
            test = {}
            for cor in correct:
                test[cor] = trans[file][cor]['truepos']

            print(file+': Recall')
            rdivDemo(test)
            test = {}
            for cor in correct:
                test[cor] = trans[file][cor]['count']
            print(file+': Cost')
            rdivDemo(test)

        set_trace()

def error_sumlatex():
    files = ['vuls_data_dom.csv', 'vuls_data_js.csv', 'vuls_data_netwerk.csv', 'vuls_data_gfx.csv',
             'vuls_data_other.csv', 'vuls_data_new.csv']
    name = {'vuls_data_dom.csv': 'Module: dom', 'vuls_data_js.csv': 'Module: js', 'vuls_data_netwerk.csv': 'Module: netwerk', 'vuls_data_gfx.csv': 'Module: gfx',
             'vuls_data_other.csv': 'Other modules', 'vuls_data_new.csv': 'Entire project', 'Median': 'Median'}
    correct = ['none','majority','machine','knee','machine3','random']
    total = {'vuls_data_dom.csv': 86, 'vuls_data_js.csv': 57, 'vuls_data_netwerk.csv': 29, 'vuls_data_gfx.csv': 28,
             'vuls_data_other.csv': 71, 'vuls_data_new.csv': 271}
    total2 = {'vuls_data_dom.csv': 3505,'vuls_data_js.csv': 1421,'vuls_data_netwerk.csv': 698,'vuls_data_gfx.csv': 4814,'vuls_data_other.csv': 18312,'vuls_data_new.csv':28750}
    metrics = ['Recall', 'Cost']


    ## Baseline
    with open("../dump/error2_new_hpcc0.pickle", 'r') as f:
        results = pickle.load(f)


    cols = ["Dataset"]+metrics
    bl = {col:[] for col in cols}
    bl0 = {}

    # trans = {'Median':{}}
    trans = results

    # for file in files:
    #     trans[file] = {}
    #     for cor in correct:
    #         trans[file][cor] = {}
    #         keys = results[file][cor][0].keys()
    #         for key in keys:
    #             trans[file][cor][key]=[]
    #             for i in xrange(len(results[file][cor])):
    #                 trans[file][cor][key].append(results[file][cor][key][i])


    medians={'recall':[],'cost':[]}
    for file in files:
        bl['Dataset'].append(name[file])
        bl0[file]={}
        # recall
        x = np.array(trans[file]['none']['truepos'])/total[file]
        medians['recall'].extend(x)
        median = str(int(np.median(x)*100))
        iqr = str(int((np.percentile(x,75)-np.percentile(x,25))*100))
        bl['Recall'].append(median+'('+iqr+')')
        bl0[file]['Recall'] = np.median(x)

        # precision
        # x = np.array(trans[file]['none']['truepos']) / (np.array(trans[file]['none']['truepos'])+np.array(trans[file]['none']['falsepos']))
        # median = str(int(np.median(x) * 100))
        # iqr = str(int((np.percentile(x, 75) - np.percentile(x, 25)) * 100))
        # bl['Precision'].append(median + '(' + iqr + ')')
        # bl0[file]['Precision'] = np.median(x)

        # cost
        x = np.array(trans[file]['none']['count']) / total2[file]
        median = str(int(np.median(x)*100))
        medians['cost'].extend(x)
        iqr = str(int((np.percentile(x,75)-np.percentile(x,25))*100))
        bl['Cost'].append(median + '(' + iqr + ')')
        bl0[file]['Cost'] = np.median(x)


    bl0['Median']={}

    bl['Dataset'].append('Median')
    median = str(int(np.median(medians['recall'])*100))
    iqr = str(int((np.percentile(medians['recall'],75)-np.percentile(medians['recall'],25))*100))
    bl['Recall'].append(median+'('+iqr+')')
    bl0['Median']['Recall'] = np.median(medians['recall'])

    median = str(int(np.median(medians['cost'])*100))
    iqr = str(int((np.percentile(medians['cost'],75)-np.percentile(medians['cost'],25))*100))
    bl['Cost'].append(median+'('+iqr+')')
    bl0['Median']['Cost'] = np.median(medians['cost'])


    df = pd.DataFrame(data=bl, columns=cols)
    df.to_csv("../dump/error_baseline.csv")

    ## all results

    cols = ["Dataset"]
    for m in metrics:
        for c in correct:
            cols.append(m+'_'+c)



    errors = ['0','10','20','30','40','50']
    for error in errors:
        with open("../dump/error2_new_hpcc"+error+".pickle", 'r') as f:
            results = pickle.load(f)

        trans = results

        # for file in files:
        #     trans[file] = {}
        #     for cor in correct:
        #         trans[file][cor] = {}
        #         keys = results[0][file][cor].keys()
        #         for key in keys:
        #             trans[file][cor][key]=[]
        #             for i in xrange(len(results)):
        #                 trans[file][cor][key].append(results[i][file][cor][key])

        dictdf = {col: [] for col in cols}

        medians={}
        for file in files:
            dictdf['Dataset'].append(name[file])
            for cor in correct:

                if not 'recall_'+cor in medians:
                    medians['recall_'+cor] = []
                if not 'cost_'+cor in medians:
                    medians['cost_'+cor] = []

                # recall
                x = np.array(trans[file][cor]['truepos']) / total[file] / bl0[file]['Recall']
                medians['recall_'+cor].extend(x)
                median = str(int(np.median(x)*100))
                iqr = str(int((np.percentile(x,75)-np.percentile(x,25))*100))
                dictdf['Recall_'+cor].append(median + '(' + iqr + ')')


                # precision
                # x = np.array(trans[file][cor]['truepos']) / (
                # np.array(trans[file][cor]['truepos']) + np.array(trans[file][cor]['falsepos'])) / bl0[file]['Precision']
                # median = str(int(np.median(x)*100))
                # iqr = str(int((np.percentile(x,75)-np.percentile(x,25))*100))
                # dictdf['Precision_'+cor].append(median + '(' + iqr + ')')


                # cost
                x = np.array(trans[file][cor]['count']) / total2[file] / bl0[file]['Cost']
                medians['cost_'+cor].extend(x)
                median = str(int(np.median(x)*100))
                iqr = str(int((np.percentile(x,75)-np.percentile(x,25))*100))
                dictdf['Cost_'+cor].append(median + '(' + iqr + ')')


        dictdf['Dataset'].append('Median')
        for cor in correct:
            median = str(int(np.median(medians['recall_'+cor])*100))
            iqr = str(int((np.percentile(medians['recall_'+cor],75)-np.percentile(medians['recall_'+cor],25))*100))
            dictdf['Recall_'+cor].append(median+'('+iqr+')')

            median = str(int(np.median(medians['cost_'+cor])*100))
            iqr = str(int((np.percentile(medians['cost_'+cor],75)-np.percentile(medians['cost_'+cor],25))*100))
            dictdf['Cost_'+cor].append(median+'('+iqr+')')


        # for file in files:
        #     dictdf['Dataset'].append(name[file])
        #     for cor in correct:
        #         # recall
        #         x = np.array(trans[file][cor]['truepos']) / total[file] / bl0[file]['Recall']
        #         median = str(int(np.median(x)*100))
        #         iqr = str(int((np.percentile(x,75)-np.percentile(x,25))*100))
        #         dictdf['Recall_'+cor].append(median + '(' + iqr + ')')
        #
        #
        #         # precision
        #         # x = np.array(trans[file][cor]['truepos']) / (
        #         # np.array(trans[file][cor]['truepos']) + np.array(trans[file][cor]['falsepos'])) / bl0[file]['Precision']
        #         # median = str(int(np.median(x)*100))
        #         # iqr = str(int((np.percentile(x,75)-np.percentile(x,25))*100))
        #         # dictdf['Precision_'+cor].append(median + '(' + iqr + ')')
        #
        #
        #         # cost
        #         x = np.array(trans[file][cor]['count']) / total2[file] / bl0[file]['Cost']
        #         median = str(int(np.median(x)*100))
        #         iqr = str(int((np.percentile(x,75)-np.percentile(x,25))*100))
        #         dictdf['Cost_'+cor].append(median + '(' + iqr + ')')


        col1 = cols[:5]+cols[7:11]
        col2 = [cols[0]]+[cols[5]]+[cols[2]]+[cols[3]]+[cols[6]]+ [cols[11]]+[cols[8]]+[cols[9]]+[cols[12]]
        df1 = pd.DataFrame(data=dictdf,
                          columns=col1)
        df1.to_csv("../dump/error_" + str(error) + '.csv')
        df2 = pd.DataFrame(data=dictdf,
                          columns=col2)
        df2.to_csv("../dump/error2_" + str(error) + '.csv')
        df = pd.DataFrame(data=dictdf,
                          columns=cols)
        df.to_csv("../dump/errorall_" + str(error) + '.csv')


############


def feature_summary():
    # import cPickle as pickle
    files = {'Arbitrary Code':28750, 'Improper Control of a Resource Through its Lifetime':28750, 'Other':28750, 'Range Error':28750, 'Code Quality':28750, 'all':28750}
    vuls = {'Arbitrary Code':118, 'Improper Control of a Resource Through its Lifetime':81, 'Other':42, 'Range Error':32, 'Code Quality':29, 'all':271}
    features = ['combine','random','text','metrics']

    result = {}
    for fea in features:
        result[fea]={f:{'x':[],'pos':[],'est':[],'stop':{0.6:[],0.7:[],0.8:[],0.85:[],0.9:[],0.95:[],0.99:[],1.0:[]}, 'stop_est':{0.9:{'recall':[],'cost':[]},0.95:{'recall':[],'cost':[]},0.99:{'recall':[],'cost':[]}}} for f in files}
        for i in xrange(30):
            filename = '../dump/features_'+fea+'_hpcc_'+str(i)+'.pickle'
            tmp = pickle.load(open(filename, 'rb'))
            for f in tmp:
                x = np.array(tmp[f]['pos']['x'])/files[f]
                pos = np.array(tmp[f]['pos']['pos'])/vuls[f]
                start = len(x)-len(tmp[f]['est']['semi'])
                est = np.array([0]*start+tmp[f]['est']['semi'])/vuls[f]
                result[fea][f]['x'].append(x)
                result[fea][f]['pos'].append(pos)
                result[fea][f]['est'].append(est)

                re={}

                for i in xrange(len(x)):
                    if pos[i]>=0.6 and not 0.6 in re:
                        re[0.6]=x[i]
                    if pos[i]>=0.7 and not 0.7 in re:
                        re[0.7]=x[i]
                    if pos[i]>=0.8 and not 0.8 in re:
                        re[0.8]=x[i]
                    if pos[i]>=0.85 and not 0.85 in re:
                        re[0.85]=x[i]
                    if pos[i]>=0.9 and not 0.9 in re:
                        re[0.9]=x[i]
                    if pos[i]>=0.95 and not 0.95 in re:
                        re[0.95]=x[i]
                    if pos[i]>=0.99 and not 0.99 in re:
                        re[0.99]=x[i]
                    if pos[i]>=1 and not 1 in re:
                        re[1]=x[i]

                for key in re:
                    result[fea][f]['stop'][key].append(re[key])

                re={}
                re2={}
                # if fea == 'metrics':
                #     continue
                for i in xrange(start,len(x)):

                    if pos[i]>=0.9*est[i] and not 0.9 in re:
                        re[0.9]=x[i]
                        re2[0.9] = pos[i]

                    if pos[i]>=0.95*est[i] and not 0.95 in re:
                        re[0.95]=x[i]
                        re2[0.95] = pos[i]
                    if pos[i]>=0.99*est[i] and not 0.99 in re:
                        re[0.99]=x[i]
                        re2[0.99] = pos[i]

                for key in re:
                    result[fea][f]['stop_est'][key]['recall'].append(re2[key])
                    result[fea][f]['stop_est'][key]['cost'].append(re[key])

    filename = '../dump/features.pickle'
    pickle.dump(result,open(filename, 'wb'))



def plot_feature():
    filename = '../dump/features.pickle'
    result = pickle.load(open(filename, 'rb'))
    files = ['Arbitrary Code', 'Improper Control of a Resource Through its Lifetime', 'Range Error', 'Code Quality']

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 30}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 28, 'axes.labelsize': 40, 'legend.frameon': True,
             'figure.autolayout': False, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)


    for file in files:
        start = 0
        end = 1000000
        for est in result['text'][file]['est']:
            start = max((start,np.where(est==0)[0][-1]+1))
            end = min((end,len(est)))
        start2 = 0
        end2 = 1000000
        for est in result['combine'][file]['est']:
            start2 = max((start2,np.where(est==0)[0][-1]+1))
            end2 = min((end2,len(est)))

        start = max((start,start2))
        end = min((end,end2))


        text = []
        for i in xrange(start,end):
            text.append([est[i] for est in result['text'][file]['est']])

        x={}
        x['cost'] = result['text'][file]['x'][0][start:end]
        x['50'] = [np.median(t) for t in text]
        x['75'] = [np.percentile(t,75) for t in text]
        x['25'] = [np.percentile(t,25) for t in text]



        combine = []
        for i in xrange(start,end):
            combine.append([est[i] for est in result['combine'][file]['est']])

        y={}
        y['cost'] = result['combine'][file]['x'][0][start:end]
        y['50'] = [np.median(t) for t in combine]
        y['75'] = [np.percentile(t,75) for t in combine]
        y['25'] = [np.percentile(t,25) for t in combine]





        plt.figure(1)
        ax=plt.subplot(111)




        ax.plot(x['cost'],x['50'],color='blue',linestyle = '-',label='Text')
        ax.plot(x['cost'],x['75'],color='blue',linestyle = '--')
        ax.plot(x['cost'],x['25'],color='blue',linestyle = '--')

        ax.plot(y['cost'],y['50'],color='red',linestyle = '-',label='Combine')
        ax.plot(y['cost'],y['75'],color='red',linestyle = '--')
        ax.plot(y['cost'],y['25'],color='red',linestyle = '--')

        ax.plot(x['cost'],[1]*len(x['cost']),color='black',linestyle = '-',label = 'true')

        plt.subplots_adjust(top=0.95, left=0.12, bottom=0.2, right=0.75)
        ax.legend(bbox_to_anchor=(1.02, 1), loc=2, ncol=1, borderaxespad=0.)

        plt.ylabel("Estimation")
        plt.xlabel("Cost")
        plt.savefig("../figure/est_" + str(file) + ".pdf")
        plt.savefig("../figure/est_" + str(file) + ".png")
        plt.close()


def sum_feature():
    filename = '../dump/features.pickle'
    result = pickle.load(open(filename, 'rb'))
    stop = [0.6,0.7,0.8,0.85,0.9,0.95,0.99,1.0]
    stop_est = [0.9,0.95,0.99]
    files = ['Arbitrary Code', 'Improper Control of a Resource Through its Lifetime', 'Range Error', 'Code Quality', 'all']
    cols = ["Dataset"]+stop



    for fea in result:
        dictdf = {col: [] for col in cols}


        medians={}
        for file in files:
            dictdf['Dataset'].append(file)
            for cor in stop:
                x = result[fea][file]['stop'][cor]
                try:
                    median = str(int(np.median(x)*100))
                except:
                    median = '0'
                try:
                    iqr = str(int((np.percentile(x,75)-np.percentile(x,25))*100))
                except:
                    iqr = '0'
                dictdf[cor].append(median + '(' + iqr + ')')

                if not cor in medians:
                    medians[cor] = []
                medians[cor].extend(x)






        dictdf['Dataset'].append('Median')
        for cor in stop:
            try:
                median = str(int(np.median(medians[cor])*100))
                iqr = str(int((np.percentile(medians[cor],75)-np.percentile(medians[cor],25))*100))
            except:
                median = '0'
                iqr = '0'
            dictdf[cor].append(median+'('+iqr+')')



        df = pd.DataFrame(data=dictdf,
                          columns=cols)
        df.to_csv("../dump/feature_stop_" + str(fea) + '.csv')







    metrics = ['recall','cost']

    cols = ["Dataset"]
    for c in stop_est:
        for m in metrics:
            cols.append(m+'_'+str(c))

    for fea in ['combine','text']:
        dictdf = {col: [] for col in cols}


        medians={}
        for file in files:
            dictdf['Dataset'].append(file)
            for cor in stop_est:
                x = result[fea][file]['stop_est'][cor]['recall']
                try:
                    median = str(int(np.median(x)*100))
                except:
                    median = '0'
                try:
                    iqr = str(int((np.percentile(x,75)-np.percentile(x,25))*100))
                except:
                    iqr = '0'
                dictdf['recall_'+str(cor)].append(median + '(' + iqr + ')')

                y = result[fea][file]['stop_est'][cor]['cost']
                try:
                    median = str(int(np.median(y)*100))
                except:
                    median = '0'
                try:
                    iqr = str(int((np.percentile(y,75)-np.percentile(y,25))*100))
                except:
                    iqr = '0'
                dictdf['cost_'+str(cor)].append(median + '(' + iqr + ')')

                if not 'recall_'+str(cor) in medians:
                    medians['recall_'+str(cor)] = []
                    medians['cost_'+str(cor)] = []
                medians['recall_'+str(cor)].extend(x)
                medians['cost_'+str(cor)].extend(y)

        dictdf['Dataset'].append('Median')
        for cor in stop_est:
            try:
                median = str(int(np.median(medians['recall_'+str(cor)])*100))
                iqr = str(int((np.percentile(medians['recall_'+str(cor)],75)-np.percentile(medians['recall_'+str(cor)],25))*100))
            except:
                median = '0'
                iqr = '0'
            dictdf['recall_'+str(cor)].append(median+'('+iqr+')')

            try:
                median = str(int(np.median(medians['cost_'+str(cor)])*100))
                iqr = str(int((np.percentile(medians['cost_'+str(cor)],75)-np.percentile(medians['cost_'+str(cor)],25))*100))
            except:
                median = '0'
                iqr = '0'
            dictdf['cost_'+str(cor)].append(median+'('+iqr+')')



        df = pd.DataFrame(data=dictdf,
                          columns=cols)
        df.to_csv("../dump/feature_est_" + str(fea) + '.csv')


if __name__ == "__main__":
    eval(cmd())
