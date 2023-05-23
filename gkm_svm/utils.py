import yaml
import numpy as np

def load_config(config_path):
    """Load a model config from a YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

# Save a list of sequences to fasta
def seq2Fasta(seqs, IDs, name="seqs"):
    file = open("{}.fa".format(name), "w")
    for i in range(len(seqs)):
        file.write(">" + IDs[i] + "\n" + seqs[i] + "\n")
    file.close()

# Get all the needed information for viz sequence of gkmexplain result. Returns importance
# scores per position along with the sequences, IDs and one-hot sequences
def get_gksvm_explain_data(explain_file, fasta_file):
    impscores = [np.array( [[float(z) for z in y.split(",")] for y in x.rstrip().split("\t")[2].split(";")]) for x in open(explain_file)]
    fasta_seqs = [x.rstrip() for (i,x) in enumerate(open(fasta_file)) if i%2==1]
    fasta_ids = [x.rstrip().replace(">", "") for (i,x) in enumerate(open(fasta_file)) if i%2==0]
    onehot_data = np.array([one_hot_encode_along_channel_axis(x) for x in fasta_seqs])
    return impscores, fasta_seqs, fasta_ids, onehot_data


# Save a list of sequences to separate pos and neg fa files. Must supply target 0 or 1 labels
def gkmSeq2Fasta(seqs, IDs, ys, name="seqs"):
    neg_mask = (ys==0)

    neg_seqs, neg_ys, neg_IDs = seqs[neg_mask], ys[neg_mask], IDs[neg_mask]
    neg_file = open("{}-neg.fa".format(name), "w")
    for i in range(len(neg_seqs)):
        neg_file.write(">" + neg_IDs[i] + "\n" + neg_seqs[i] + "\n")
    neg_file.close()

    pos_seqs, pos_ys, pos_IDs = seqs[~neg_mask], ys[~neg_mask], IDs[~neg_mask]
    pos_file = open("{}-pos.fa".format(name), "w")
    for i in range(len(pos_seqs)):
        pos_file.write(">" + pos_IDs[i] + "\n" + pos_seqs[i] + "\n")
    pos_file.close()

    




