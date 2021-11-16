import argparse
import json
from typing import Tuple, List

import cv2
import pandas as pd
import editdistance
from path import Path

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor


class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = '../model/charList.txt'
    fn_summary = '../model/summary.json'
    fn_corpus = '../data/corpus.txt'


def get_img_height() -> int:
    """Fixed height for NN."""
    return 50


def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    return 250, get_img_height()


def write_summary(char_error_rates: List[float], word_accuracies: List[float]) -> None:
    """Writes training summary file for NN."""
    with open(FilePaths.fn_summary, 'w') as f:
        json.dump({'charErrorRates': char_error_rates, 'wordAccuracies': word_accuracies}, f)


def train(model: Model,
          loader: DataLoaderIAM,
          line_mode: bool,
          early_stopping: int = 25) -> None:
    """Trains NN."""
    epoch = 0  # number of training epochs since start
    summary_char_error_rates = []
    summary_word_accuracies = []
    preprocessor = Preprocessor(get_img_size(line_mode), data_augmentation=True, line_mode=line_mode)
    best_char_error_rate = float('inf')  # best valdiation character error rate
    no_improvement_since = 0  # number of epochs no improvement of character error rate occurred
    # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.train_set()
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            batch = loader.get_next()
            batch = preprocessor.process_batch(batch)
            loss = model.train_batch(batch)
            print(f'Epoch: {epoch} Batch: {iter_info[0]}/{iter_info[1]} Loss: {loss}')

        # validate
        char_error_rate, word_accuracy = validate(model, loader, line_mode)

        # write summary
        summary_char_error_rates.append(char_error_rate)
        summary_word_accuracies.append(word_accuracy)
        write_summary(summary_char_error_rates, summary_word_accuracies)

        # if best validation accuracy so far, save model parameters
        min_delta=0.0001
        if (char_error_rate + min_delta) <= best_char_error_rate:
            print('Character error rate improved, save model')
            best_char_error_rate = char_error_rate
            no_improvement_since = 0
            model.save()
        else:
            print(f'Character error rate not improved, best so far: {best_char_error_rate * 100.0}%')
            no_improvement_since += 1

        # stop training if no more improvement in the last x epochs
        if no_improvement_since >= early_stopping:
            print(f'No more improvement since {early_stopping} epochs. Training stopped.')
            break


def validate(model: Model, loader: DataLoaderIAM, line_mode: bool) -> Tuple[float, float]:
    """Validates NN."""
    print('Validate NN')
    loader.validation_set()
    preprocessor = Preprocessor(get_img_size(line_mode), line_mode=line_mode)
    num_char_err = 0
    num_char_total = 0
    num_word_ok = 0
    num_word_total = 0
    while loader.has_next():
        iter_info = loader.get_iterator_info()
        print(f'Batch: {iter_info[0]} / {iter_info[1]}')
        batch = loader.get_next()
        batch = preprocessor.process_batch(batch)
        recognized, _ = model.infer_batch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            num_word_ok += 1 if batch.gt_texts[i] == recognized[i] else 0
            num_word_total += 1
            dist = editdistance.eval(recognized[i], batch.gt_texts[i])
            num_char_err += dist
            num_char_total += len(batch.gt_texts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gt_texts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation result
    char_error_rate = num_char_err / num_char_total
    word_accuracy = num_word_ok / num_word_total
    print(f'Character error rate: {char_error_rate * 100.0}%. Word accuracy: {word_accuracy * 100.0}%.')
    return char_error_rate, word_accuracy


def infer(model: Model, fn_img: Path) -> None:
    """Recognizes text in image provided by file path."""
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)
    #print(f'Recognized: "{recognized[0]}"')
    #print(f'Probability: {probability[0]}')
    return recognized[0]

def main():
    """Main function."""

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', choices=['train','test'], default='train')
    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath')
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=16)
    parser.add_argument('--test_img', help='Testing images ', required=False)
    parser.add_argument('--test_csv', help='Testing map ', required=False)
    parser.add_argument('--validate_img', help='Validation images ',  required=False)
    parser.add_argument('--validate_csv', help='Validation map ',  required=False)
    parser.add_argument('--train_img', help='Training images ',  required=False)
    parser.add_argument('--train_csv', help='Training map ', required=False)
    parser.add_argument('--line_mode', help='Train to read text lines instead of single words.', action='store_true')
    parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=10)
    parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')
    args = parser.parse_args()
    
    # set chosen CTC decoder
    decoder_mapping = {'bestpath': DecoderType.BestPath,
                       'beamsearch': DecoderType.BeamSearch,
                       'wordbeamsearch': DecoderType.WordBeamSearch}
    decoder_type = decoder_mapping[args.decoder]

    # train or validate on IAM dataset
    if args.mode=='train':
        # load training data, create TF model
        loader = DataLoaderIAM(args.train_img,args.train_csv,args.validate_img,args.validate_csv,args.batch_size)
        char_list = loader.char_list

        # when in line mode, take care to have a whitespace in the char list
        if args.line_mode and ' ' not in char_list:
            char_list = [' '] + char_list

        # save characters of model for inference mode
        open(FilePaths.fn_char_list, 'w').write(''.join(char_list))

        # save words contained in dataset into file
        open(FilePaths.fn_corpus, 'w').write(' '.join(loader.train_words + loader.validation_words))

        model = Model(char_list, decoder_type)
        train(model, loader, line_mode=args.line_mode, early_stopping=args.early_stopping)


    # test the saved model on a set of images
    elif args.mode == 'test':
        model = Model(list(open(FilePaths.fn_char_list).read()), decoder_type, must_restore=True, dump=args.dump)
        map_path = './dataset/' + args.test_csv
        img_path = './dataset/' + args.test_img
        #assert map_path.exists()
        #ssert img_path.exists()
        num_char_err = 0
        num_char_total = 0
        file = pd.read_csv(map_path)
        image_name_list = list(file['Image'])
        gt_list = list(file['Word'])
        total_words = 0
        words_ok = 0
        for i in range(len(image_name_list)):
            total_words = total_words + 1
            recognised_string = infer(model, img_path + image_name_list[i])
            if gt_list[i] == recognised_string:
                words_ok = words_ok + 1
            dist = editdistance.eval(recognised_string, gt_list[i])
            #print(" recognised  and ground truth",recognised_string, gt_list[i])
            num_char_err += dist
            num_char_total += len(gt_list[i])
            #print("Ground Truth : ",gt_list[i])
        print("Word Accuracy : ",round((words_ok/total_words)*100,2))
        print("Character Error rate : ",round((num_char_err/num_char_total)*100,2))

if __name__ == '__main__':
    main()
