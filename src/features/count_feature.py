import pickle

from src import config
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
                    list(map(try_divide, df["count_of_unique_%s_%s" % (feat_name, gram)],
                             df["count_of_%s_%s" % (feat_name, gram)]))

        # number of sentences in text
        for feat_name in feat_names:
            df['len_sent_%s' % feat_name] = df[feat_name].apply(lambda x: len(tokenizer(x)))

        # dump the basic counting features into a file
        feat_names = [n for n in df.columns if "count" in n or "ratio" in n or "len_sent" in n]

        print('CountFeatures:', df.head())
        # split into train, test portion and save in separate files
        train = df[df['type'] == 'train']
        print('train:')
        print(train[['text', 'id', 'count_of_text_unigram']].head())
        count_feature_train = train[feat_names].values
        count_feature_train_path = config.output_dir + "train.count.pkl"
        with open(count_feature_train_path, "wb") as f:
            pickle.dump(feat_names, f)
            pickle.dump(count_feature_train, f)
        print('count features for training saved in %s' % count_feature_train_path)

        test = df[df['type'] == 'test']
        print('test:')
        print(test[['text', 'id', 'count_of_text_unigram']].head())
        if test.shape[0] > 0:
            # test set exists
            print('saving test set')
            count_feature_test = test[feat_names].values
            count_feature_test_path = config.output_dir + "test.count.pkl"
            with open(count_feature_test_path, 'wb') as f:
                pickle.dump(feat_names, f)
                pickle.dump(count_feature_test, f)
                print('count features for test saved in %s' % count_feature_test_path)

    def read(self, header='train'):
        path = config.output_dir + "%s.count.pkl" % header
        with open(path, "rb") as f:
            feat_names = pickle.load(f)
            count_feature = pickle.load(f)
            print('feature names: ', feat_names)
            print('count_feature.shape:', count_feature.shape)
        return [count_feature]

    def official_news(self):
        official_names = ["text"]
        print("generate official news features")