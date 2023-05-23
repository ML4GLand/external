def get_scores(fname):
    f = open(fname)
    d = [float(x.strip().split('\t')[1]) for x in f]
    f.close()
    return d

# Function to score test predictions made on the positive and negative classes
# pos_file => ls-gkm scores for positive test seqs
# neg_file => ls-gkm scores for negative test seqs
# thresh => float threshold for accuracy scoring
def score(pos_file, neg_file, thresh):
    pos_scores = get_scores(pos_file)
    neg_scores = get_scores(neg_file)
    labels = [1]*len(pos_scores) + [0]*len(neg_scores)

    labels_shuf = deepcopy(labels)
    shuffle(labels_shuf)

    auprc = average_precision_score(labels, pos_scores+neg_scores)
    auroc = roc_auc_score(labels, pos_scores+neg_scores)
    auprc_shuf = average_precision_score(labels_shuf, pos_scores+neg_scores)
    auroc_shuf = roc_auc_score(labels_shuf, pos_scores+neg_scores)
    acc_thresh0 = sum([x==int(y>thresh) for x,y in zip(labels, pos_scores+neg_scores)])/len(labels)
    acc_thresh0_shuf = sum([x==int(y>thresh) for x,y in zip(labels_shuf, pos_scores+neg_scores)])/len(labels)

    print("Metric\tValue\tRandomised")
    print("Accuracy_at_threshold_{}\t{:.4f}\t{:.4f}".format(thresh, acc_thresh0, acc_thresh0_shuf))
    print("AUROC\t{:.4f}\t{:.4f}".format(auroc, auroc_shuf))
    print("AUPRC\t{:.4f}\t{:.4f}".format(auprc, auprc_shuf))