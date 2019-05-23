import tensorflow as tf
import pickle
from cnn_sm.model.net import SenCNN
from cnn_sm.model.utils import PreProcessor
from pathlib import Path
from konlpy.tag import Okt
from tqdm import tqdm
from absl import app
import sys
sys.path.append('..')


def create_dataset(filepath, batch_size, shuffle=True, drop_remainder=True):
    ds = tf.data.TextLineDataset(filepath)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    ds = ds.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return ds

def main(argv):
    test_filepath = Path.cwd()/'data'/'test.txt'

    with open(Path.cwd()/'data/vocab.pkl', mode='rb') as f:
        vocab = pickle.load(f)

    test_ds = create_dataset(str(test_filepath), 128, shuffle=False)  # 평가 데이터는 셔플 ㄴㄴ
    # params = json.load((proj_dir/cfgpath).open())

    tokenized = Okt()
    pre_processor = PreProcessor(vocab=vocab, tokenizer=tokenized)

    # create model
    model = SenCNN(num_classes=2, vocab=vocab)

    # create optimizer & loss_fn
    epochs = 10
    learning_rate = 1e-3

    opt = tf.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.losses.SparseCategoricalCrossentropy()

    # metrics
    test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # training

    for epoch in tqdm(range(epochs), desc='epochs'):
        # tr_loss

        tf.keras.backend.set_learning_phase(0) # test mode
        for _, mb in tqdm(enumerate(test_ds), desc='steps'):
            x_mb, y_mb = pre_processor.convert2idx(mb)
            mb_loss = loss_fn(y_mb, model(x_mb))

            test_loss_metric.update_state(mb_loss)
            test_accuracy_metric.update_state(y_mb, model(x_mb))

        test_mean_loss = test_loss_metric.result()
        test_mean_accuracy = test_accuracy_metric.result()


        tqdm.write('epoch : {}, test_accuracy : {:.3f}, test_loss : {:.3f}'.format(epoch + 1, test_mean_accuracy, test_mean_loss))

if __name__ == "__main__":
    app.run(main)