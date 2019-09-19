import pickle

from src.features.math_util import try_divide
from src.features.tokenizer import tokenizer


class CountFeatureGenerator(object):
    def __init__(self, name='countFeatureGenerator'):
        self.name = name

    def process(self, df):
        grams = ["unigram", "bigram", "trigram"]
        feat_names = ["text"]
        print("generate counting features")
        for feat_name in feat_names:
            for gram in grams:
                df["count_of_%s_%s" % (feat_name, gram)] = list(
                    df.apply(lambda x: len(x[feat_name + "_" + gram]), axis=1))
                df["count_of_unique_%s_%s" % (feat_name, gram)] = \
                    list(df.apply(lambda x: len(set(x[feat_name + "_" + gram])), axis=1))
                df["ratio_of_unique_%s_%s" % (feat_name, gram)] = \
                    map(try_divide, df["count_of_unique_%s_%s" % (feat_name, gram)],
                        df["count_of_%s_%s" % (feat_name, gram)])

        # number of sentences in text
        for feat_name in feat_names:
            df['len_sent_%s' % feat_name] = df[feat_name].apply(lambda x: len(tokenizer(x)))

        # dump the basic counting features into a file
        feat_names = [n for n in df.columns if "count" in n or "ratio" in n or "len_sent" in n]

        print('BasicCountFeatures:', df)

        # split into train, test portion and save in separate files
        train = df[df['type'] == 'train']
        print('train:')
        print(train[['text', 'id', 'count_of_text_unigram']].head())
        xBasicCountsTrain = train[feat_names].values
        outfilename_bcf_train = "train.basic.pkl"
        with open(outfilename_bcf_train, "wb") as f:
            pickle.dump(feat_names, f)
            pickle.dump(xBasicCountsTrain, f)
        print('basic counting features for training saved in %s' % outfilename_bcf_train)

        test = df[df['type'] == 'test']
        print('test:')
        print(test[['text', 'id', 'count_of_text_unigram']].head())
        if test.shape[0] > 0:
            # test set exists
            print('saving test set')
            xBasicCountsTest = test[feat_names].values
            outfilename_bcf_test = "test.basic.pkl"
            with open(outfilename_bcf_test, 'wb') as f:
                pickle.dump(feat_names, f)
                pickle.dump(xBasicCountsTest, f)
                print('basic counting features for test saved in %s' % outfilename_bcf_test)

    def read(self, header='train'):
        filename_bcf = "%s.basic.pkl" % header
        with open(filename_bcf, "rb") as f:
            feat_names = pickle.load(f)
            xBasicCounts = pickle.load(f)
            print('feature names: ', feat_names)
            print('xBasicCounts.shape:', xBasicCounts.shape)
        return [xBasicCounts]


if __name__ == '__main__':
    import pandas as pd
    from src import config

    df_all = pd.read_pickle(config.data_file_path)
    cf = CountFeatureGenerator()
    cf.process(df_all)
    cf.read()
