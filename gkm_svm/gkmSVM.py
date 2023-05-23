from ..model import Model

class gkmSVM(Model):

    def __init__(
        self, 
        prefix='./gkmSVM', 
        word_length=11, 
        mismatches=3, 
        C=1,
        threads=1, 
        cache_memory=100, 
        verbosity=4
    ):
        self.word_length = word_length
        self.mismatches = mismatches
        self.C = C
        self.threads = threads
        self.prefix = '_'.join(map(str, (prefix, word_length, mismatches, C)))
        options_list = zip(
            ['-l', '-d', '-c', '-T', '-m', '-v'],
            map(str, (word_length, mismatches, C, threads, cache_memory, verbosity)))
        self.options = ' '.join([' '.join(option) for option in options_list])

    @property
    def model_file(self):
        model_fname = '{}.model.txt'.format(self.prefix)
        return model_fname if os.path.isfile(model_fname) else None

    @staticmethod
    def encode_sequence_into_fasta_file(sequence_iterator, ofname):
        """writes sequences into fasta file
        """
        with open(ofname, "w") as wf:
            for i, seq in enumerate(sequence_iterator):
                print('>{}'.format(i), file=wf)
                print(seq, file=wf)

    def train(self, X, y, validation_data=None):
        """
        Trains gkm-svm, saves model file.
        """
        y = y.squeeze()
        pos_sequence = X[y]
        neg_sequence = X[~y]
        pos_fname = "%s.pos_seq.fa" % self.prefix
        neg_fname = "%s.neg_seq.fa" % self.prefix
        # create temporary fasta files
        self.encode_sequence_into_fasta_file(pos_sequence, pos_fname)
        self.encode_sequence_into_fasta_file(neg_sequence, neg_fname)
        # run command
        command = ' '.join(
            ('gkmtrain', self.options, pos_fname, neg_fname, self.prefix))
        process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        process.wait()  # wait for it to finish
        # remove fasta files
        os.system("rm %s" % pos_fname)
        os.system("rm %s" % neg_fname)

    def predict(self, X):
        if self.model_file is None:
            raise RuntimeError("GkmSvm hasn't been trained!")
        # write test fasta file
        test_fname = "%s.test.fa" % self.prefix
        self.encode_sequence_into_fasta_file(X, test_fname)
        # test gkmsvm
        temp_ofp = tempfile.NamedTemporaryFile()
        threads_option = '-T %s' % (str(self.threads))
        command = ' '.join(['gkmpredict',
                            test_fname,
                            self.model_file,
                            temp_ofp.name,
                            threads_option])
        process = subprocess.Popen(command, shell=True)
        process.wait()  # wait for it to finish
        os.system("rm %s" % test_fname)  # remove fasta file
        # get classification results
        temp_ofp.seek(0)
        y = np.array([line.split()[-1] for line in temp_ofp], dtype=float)
        temp_ofp.close()
        return np.expand_dims(y, 1)