from absl import flags

FLAGS = flags.FLAGS

# flags.DEFINE_string('input_data', 'data_in', 'input data folder')
# flags.DEFINE_string('model_data', 'nsmc-master', 'model data folder')
# flags.DEFINE_string('full_data', 'ratings.txt', 'train data')
# flags.DEFINE_string('train_data', 'ratings_train.txt', 'train data')
# flags.DEFINE_string('test_data', 'ratings_test.txt', 'test data')
#
# flags.DEFINE_string('train_npy', 'train.npy', 'train data')
# flags.DEFINE_string('train_label_npy', 'train_label.npy', 'train data')
# flags.DEFINE_string('test_npy', 'test.npy', 'test data')
# flags.DEFINE_string('test_label_npy', 'test_label.npy', 'test data')
#
# flags.DEFINE_string('train_text', 'train.txt', 'train data')
# flags.DEFINE_string('eval_text', 'eval.txt', 'eval data')
# flags.DEFINE_string('test_text', 'test.txt', 'test data')
#
# flags.DEFINE_string('vocab', 'vocab.pkl', 'vocab data')

flags.DEFINE_integer('classes', 2, 'classes')
flags.DEFINE_integer('length', 70, 'max length')
flags.DEFINE_integer('epochs', 1, 'epochs')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')