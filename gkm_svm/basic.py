import os
import subprocess

def train_clf_with_cv(
    positives, 
    negatives, 
    out_prefix, 
    num_folds=5, 
    word_len=10, 
    info_pos=6, 
    max_mis=3, 
    seed=None,
    strand=False
):
    """Train the SVM with num_folds cross-validation, then make ROC and PR curves.

    Parameters
    ----------
    positives : str
        Name of FASTA file with positives.
    negatives : str
        Name of FASTA file with negatives.
    out_prefix : str
        The prefix to use for files corresponding to the SVM.
    num_folds : int
        The number of folds to use in cross-validation.
    word_len : int
        The length of a word (l).
    info_pos : int
        The number of informative positions (k).
    max_mis : int
        The maximum number of mismatches allowed (m), m <= l - k.
    seed : int or None
        If specified, seed how the data is split for CV. If None, don't seed the split.
    strand : bool
        If True, reverse compliments will NOT be considered.

    Returns
    -------
    figs : (figure handle, figure handle)
        Handles to the figure for training ROC and PR curves.
    fpr_mean : np.array
        The FPR/recall values to use on the x-axis for ROC and PR curves.
    tpr_list : np.array, shape = [num_folds, len(fpr_mean)]
        The TPR at each value of fpr_mean for each fold of the data.
    precision_list : np.array, shape = [num_folds, len(fpr_mean)]
        The precision at each value of fpr_mean for each fold of the data.
    f_list : np.array, shape = [num_folds, ]
        The F-beta score for each fold of the CV
    cv_scores : pd.DataFrame
        The predictions of each sequence when in the validation set.
    """
    # Read in positive and negatives, and then join together for cross-validation.
    positive_ser = fasta_seq_parse_manip.read_fasta(positives)
    negative_ser = fasta_seq_parse_manip.read_fasta(negatives)
    sequences_ser = positive_ser.append(negative_ser)
    labels_ser = sequences_ser.isin(positive_ser)
    positive_freq = labels_ser.sum() / labels_ser.size

    # Temp directory to write stuff for folds
    tmp_out_dir = os.path.join(os.getcwd(), f"_gkmsvmCvTmp_{word_len}_{info_pos}_{max_mis}")
    run_subprocess(["mkdir", tmp_out_dir])

    # Set up the cross-validation.
    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    fpr_mean = np.linspace(0, 1, 100)
    tpr_list = []
    precision_list = []
    f_list = []
    cv_scores = pd.DataFrame([], columns=["score", "fold"])

    # For each fold
    for i, (train_idx, val_idx) in enumerate(folds.split(sequences_ser, labels_ser), 1):
        logger.info(f"Now running on fold {i}")

        # Get the training data, separate into positives and negatives, and write to file.
        train_seqs = sequences_ser[train_idx]
        train_labels = labels_ser[train_idx]
        train_positives = train_seqs[train_labels]
        train_negatives = train_seqs[~train_labels]
        train_pos_file = os.path.join(tmp_out_dir, f"positives{i}.fasta")
        train_neg_file = os.path.join(tmp_out_dir, f"negatives{i}.fasta")
        fasta_seq_parse_manip.write_fasta(train_positives, train_pos_file)
        fasta_seq_parse_manip.write_fasta(train_negatives, train_neg_file)

        # Train the SVM on this fold of training data
        fold_prefix = os.path.join(tmp_out_dir, f"Fold{i}")
        train_svm(train_pos_file, train_neg_file, fold_prefix, word_len, info_pos, max_mis, strand=strand)

        # Now get the validation data and write to file
        val_seqs = sequences_ser[val_idx]
        val_labels = labels_ser[val_idx]
        val_filename = os.path.join(tmp_out_dir, f"validation{i}.fasta")
        fasta_seq_parse_manip.write_fasta(val_seqs, val_filename)

        # Make predictions, then compute the ROC and PR curves
        tpr, precision, scores, f_beta = predict_and_eval(val_filename, val_labels, fold_prefix, word_len, info_pos,
                                                          max_mis, fpr_mean, strand=strand)
        tpr_list.append(tpr)
        precision_list.append(precision)
        f_list.append(f_beta)

        # Make the scores a df, add the fold information
        scores = scores.to_frame(name="score")
        scores["fold"] = i
        cv_scores = cv_scores.append(scores)

    # Get rid of temporary files and train the SVM on all the data
    run_subprocess(["rm", "-r", tmp_out_dir])
    logger.info("Now training on full dataset")
    train_svm(positives, negatives, out_prefix, word_len, info_pos, max_mis, strand=strand)
    run_subprocess(["rm", "f{out_prefix}.kernel"])

    # Plot ROC and PR curves
    tpr_list = np.array(tpr_list)
    precision_list = np.array(precision_list)
    f_list = np.array(f_list)
    figs, _, _, _, _ = plot_utils.roc_pr_curves(fpr_mean,
                                                [tpr_list],
                                                [precision_list],
                                                [f"{word_len}mer, {word_len - info_pos} gaps"],
                                                model_colors=["black"],
                                                prc_chance=positive_freq,
                                                figname=f"{out_prefix}_training")

    return figs, fpr_mean, tpr_list, precision_list, f_list, cv_scores

def train_reg_with_cv():
    pass

def score_all_kmers(word_len, info_pos, max_mis, svm_prefix, out_prefix):
    """Generate all k-mers of length word_len and score them against a trained SVM

    Parameters
    ----------
    word_len : int
        The length of a word (l).
    info_pos : int
        The number of informative positions (k).
    max_mis : int
        The maximum number of mismatches allowed (m), m <= l - k.
    svm_prefix : str
        The prefix for SVM files.
    out_prefix : str
        The prefix to use for k-mer files.

    Returns
    -------
    scores : pd.Series
        The scores assigned to every k-mer.

    """
    kmer_fasta = f"{out_prefix}.fasta"
    # Generate all possible k-mers
    alphabet = "ACGT"
    kmers = pd.Series(["".join(i) for i in itertools.product(alphabet, repeat=word_len)])
    kmers.index = kmers.values
    fasta_seq_parse_manip.write_fasta(kmers, kmer_fasta)

    # Make predictions
    logger.info("Scoring k-mers")
    scores = predict(kmer_fasta, svm_prefix, out_prefix, word_len=word_len, info_pos=info_pos, max_mis=max_mis)
    scores = scores.sort_values(ascending=False)
    logger.info("Finished scoring k-mers")
    return scores

def main(positives, negatives, out_prefix, num_folds=5, word_len=10, info_pos=6, max_mis=3, seed=None,
         predictions=None, score_kmers=True, strand=False):
    """Train an SVM with CV, plot ROC and PR curves with AUC scores, train final SVM on full dataset, and optionally
    make predictions on independent datasets.

    Parameters
    ----------
    positives : str
        The FASTA file of positive examples.
    negatives : str
        The FASTA file of negative examples.
    out_prefix : str
        The prefix to use for files corresponding to the SVM.
    num_folds : int
        The number of folds to use in cross-validation.
    word_len : int
        The length of a word (l).
    info_pos : int
        The number of informative positions (k).
    max_mis : int
        The maximum number of mismatches allowed (m), m <= l - k.
    seed : int or None
        If specified, seed how the data is split for CV. If None, don't seed the split.
    predictions : list[str] or None
        If specified, a list of FASTA files to make predictions on. Each file will be used as an independent set of
        predictions after training the final model.
    score_kmers : bool
        If True, generate all possible k-mers and score them against the SVM.
    strand : bool
        If True, reverse compliments will NOT be considered.

    Returns
    -------
    figs : (figure handle, figure handle)
        Handles to the figures for training ROC and PR curves.
    fpr_mean : np.array
        The FPR/recall values to use on the x-axis for ROC and PR curves.
    tpr_list : np.array, shape = [num_folds, len(fpr_mean)]
        The TPR at each value of fpr_mean for each fold of the training data.
    precision_list : np.array, shape = [num_folds, len(fpr_mean)]
        The precision at each value of fpr_mean for each fold of the training data.
    f_list : np.array, shape = [num_folds, ]
        The F-beta score for each fold of the CV.
    prediction_values : list[pd.Series]
        If predictions is specified, each value of the list is a pd.Series that is the score assigned to each
        sequence in the dataset.
    kmer_scores : pd.Series or None
        If specified, the score of every k-mer (length word_len) against the trained SVM.

    """
    # Train the SVM
    figs, fpr_mean, tpr_list, precision_list, f_list = train_with_cv(
        positives, negatives, out_prefix, num_folds=num_folds, word_len=word_len, info_pos=info_pos, max_mis=max_mis,
        seed=seed, strand=strand
    )

    # Make predictions on each provided set
    prediction_values = []
    if predictions:
        for file in predictions:
            logger.info(f"Making predictions on {file}")
            _, prefix = os.path.split(file)
            prefix, _ = prefix.split(".")
            prediction_values.append(predict(file, out_prefix, f"{out_prefix}_{prefix}", word_len, info_pos, max_mis))

    # Score k-mers, if desired
    path, _ = os.path.split(out_prefix)
    if score_kmers:
        kmer_scores = score_all_kmers(word_len, info_pos, max_mis, out_prefix, os.path.join(path, f"all{word_len}mers"))
    else:
        kmer_scores = None

    return figs, fpr_mean, tpr_list, precision_list, f_list, prediction_values, kmer_scores


if __name__ == "__main__":
    # Setup console logging
    console = logging.StreamHandler()
    allLoggers = logging.getLogger()
    allLoggers.setLevel(logging.INFO)
    allLoggers.addHandler(console)
    log_format = "[%(asctime)s][%(levelname)-7s] %(message)s"
    log_formatter = logging.Formatter(log_format, "%Y-%m-%d %H:%M:%S")
    console.setFormatter(log_formatter)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("positives", metavar="positives.fasta", help="FASTA file of the positive sequences.")
    parser.add_argument("negatives", metavar="negatives.fasta", help="FASTA file of the negative sequences.")
    parser.add_argument("out_prefix", help="The path and prefix to use for output. The kernel will be written to "
                                          "out_prefix.kernel, support vectors will be output to out_prefix_svseq.fa, "
                                          "and support vector weights will be output to out_prefix_svalpha.out.")
    parser.add_argument("--folds", type=int, default=5, help="The number of folds to use in cross-validation.")
    parser.add_argument("--word_len", type=int, default=10, help="The total length of words to use (l).")
    parser.add_argument("--info_pos", type=int, default=6, help="The number of informative positions to use (k).")
    parser.add_argument("--max_mis", type=int, default=3, help="The maximum number of mismatches allowed (m), m <= l-k.")
    parser.add_argument("--seed", type=int, help="Seed for splitting data. If unspecified, no seeding is done.")
    parser.add_argument("--predictions", metavar="predictionSet.fasta", nargs="+",
                        help="After training a model, make predictions on each file separately.")
    parser.add_argument("--score_kmers", action="store_true", help="If specified, generate all possible l-mers and score them against the final SVM.")
    parser.add_argument("--strand", action="store_true", help="If specified, reverse compliments will NOT be "
                                                              "considered.")
    args = parser.parse_args()
    main(args.positives, args.negatives, args.out_prefix, num_folds=args.folds, word_len=args.word_len,
         info_pos=args.info_pos, max_mis=args.max_mis, seed=args.seed, predictions=args.predictions,
         score_kmers=args.score_kmers, strand=args.strand)