import tensorflow as tf
import pickle
import fire
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

def get_accuracy(model, dataset, preprocess_fn):
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    if tf.keras.backend.learning_phase():
        tf.keras.backend.set_learning_phase(0)

    for step, mb in tqdm(enumerate(dataset), desc='steps'):
        x_mb, y_mb = preprocess_fn.convert2idx(mb)
        x_mb = preprocess_fn.pad_sequences(x_mb, 70)
        x_mb, y_mb = preprocess_fn.convert_to_tensor(x_mb, y_mb)
        score_mb = model(x_mb)

        accuracy_metric.update_state(y_mb, score_mb)
    
    mean_accuracy = accuracy_metric.result()

    return mean_accuracy


def main():
    proj_dir = Path.cwd()
    # tr_filepath = Path.cwd() / 'data' / 'train.txt'
    # val_filepath = Path.cwd() / 'data' / 'val.txt'
    test_filepath = Path.cwd() / 'data' / 'test.txt'

    with open(Path.cwd() / 'data/vocab.pkl', mode='rb') as f:
        vocab = pickle.load(f)

    # create dataset
    # tr_ds = create_dataset(str(tr_filepath), 128, False, False)
    # val_ds = create_dataset(str(val_filepath), 128, False, False)
    test_ds = create_dataset(str(test_filepath), 128, False, False)


    tokenized = Okt()
    pre_processor = PreProcessor(vocab=vocab, tokenizer=tokenized)

    # create model
    model = SenCNN(num_classes=2, vocab=vocab)
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(save_path=tf.train.latest_checkpoint(proj_dir/'checkpoint'))

    # evaluation
    # tr_acc = get_accuracy(model, tr_ds, pre_processor.convert2idx)
    # val_acc = get_accuracy(model, val_ds, pre_processor.convert2idx)
    test_acc = get_accuracy(model, test_ds, pre_processor)
    
    print('test_acc: {:.2%}'.format(test_acc))


if __name__ == "__main__":
    fire.Fire(main)