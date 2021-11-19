from collections import OrderedDict, namedtuple

para_dict = OrderedDict()

para_dict['batch_size'] = 512
para_dict['es_min_delta'] = 0.0
para_dict['es_patience'] = 5
para_dict['log_path'] = "tensorboard/han_voc"
para_dict['lr'] = 0.01
para_dict['momentum'] = 0.9
para_dict['num_epoches'] = 1
para_dict['saved_path'] = "trained_models"
para_dict['sent_hidden_size'] = 50
para_dict['test_interval'] = 5
para_dict['test_set'] = "/Users/hupidong/Work/data/yelp/yelp_review_polarity/test.csv"
para_dict['train_set'] = "/Users/hupidong/Work/data/yelp/yelp_review_polarity/test.csv"
para_dict['word2vec_path'] = "/Users/hupidong/Work/data/word-embeddings/glove.6B/glove.6B.50d.txt"
para_dict['word_hidden_size'] = 50

PARA = namedtuple("PARA", ["batch_size",
                           "es_min_delta",
                           "es_patience",
                           "log_path",
                           "lr",
                           "momentum",
                           "num_epoches",
                           "saved_path",
                           "sent_hidden_size",
                           "test_interval",
                           "test_set",
                           "train_set",
                           "word2vec_path",
                           "word_hidden_size"])
MY_PARAMETERS = PARA._make(para_dict.values())
