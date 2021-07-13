# -*- coding: utf-8 -*- #

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
import torchaudio as ta
compute_fbank = ta.compliance.kaldi.fbank


def extract_feature(input_file, feature='fbank', dim=40, delta=False, delta_delta=False, cmvn=True, save_feature=None):

    y, sr = ta.load_wav(input_file)

    if feature == 'fbank':
        feat = compute_fbank(y, num_mel_bins=dim, sample_frequency=sr)
    else:
        raise ValueError('Unsupported Acoustic Feature: ' + feature)

    # Apply delta
    # feat = [feat]
    # if delta and feature != 'linear':
    #     feat.append(librosa.feature.delta(feat[0]))
    # if delta_delta and feature != 'linear':
    #     feat.append(librosa.feature.delta(feat[0], order=2))
    # feat = np.concatenate(feat, axis=0)

    if cmvn:
        feat = (feat - feat.mean(axis=1)[:,np.newaxis]) / (feat.std(axis=1)+1e-16)[:,np.newaxis]
    if save_feature is not None:
        np.save(save_feature,feat.numpy())
        return len(feat)
    else:
        return np.swapaxes(feat, 0, 1).astype('float32')


def boolean_string(s):
    if s not in ['False', 'True']:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def acoustic_preprocess(args, dim, output_dir):

    if output_dir == 'None':
        output_dir = args.output_path
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    file_list = []
    with open(args.scp, 'r', encoding='utf-8') as fid:
        for line in fid:
            idx, path = line.strip().split()
            file_list.append([idx, path])

    print('Extracting acoustic feature...', flush=True)
    tr_x = Parallel(n_jobs=args.n_jobs)(
     delayed(extract_feature)
      (str(file),feature=args.feature_type, dim=dim, delta=args.delta, delta_delta=args.delta_delta, cmvn=args.apply_cmvn,
       save_feature=os.path.join(output_dir, idx+'.npy')
      ) for idx, file in tqdm(file_list)
    )

    print('All done, saved at', output_dir, 'exit.')


def generate_feat_file(root, dir_path, type='train'):
    feats_files = {}
    speakers_files = {}
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(root, type)):
        for filename in filenames:
            if filename.endswith('.npy'):
                filepath = os.sep.join([dirpath, filename])
                feats_files[filename.split('.')[0]] = filepath
                # speaker info
                speakers_files[filename.split('.')[0]] = filename[7:11]

    dir_path = os.path.join(dir_path, type)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(os.path.join(dir_path, 'feats.scp'), mode='w') as f:
        for k in feats_files.keys():
            f.write(k + ' ' + feats_files[k] + '\n')
    with open(os.path.join(dir_path, 'speakers.scp'), mode='w') as f:
        for k in speakers_files.keys():
            f.write(k + ' ' + speakers_files[k] + '\n')
    return speakers_files


def generate_transcript(transcript_path='/home/mlxu/dataall/Ai-shell/aishell_transcript_v0.8.txt'):
    targets_dict = {}
    with open(transcript_path, 'r', encoding='utf-8') as t:
        for line in t:
            parts = line.strip().split()
            utt_id = parts[0]
            targets_dict[utt_id] = ' '.join(list(''.join(parts[1:])))
    return targets_dict


def main():
    parser = argparse.ArgumentParser(description='preprocess arguments for Aishell1 dataset.')

    parser.add_argument('--scp', default='/home/mlxu/PycharmProjects/neural_sp/data/eval/wav.scp',
                        type=str)
    parser.add_argument('--output_path', default='/home/mlxu/dataall/Aishell_npy/eval_80', type=str,
                        help='Path to store output', required=False)
    parser.add_argument('--feature_type', default='fbank', type=str,
                        help='Feature type ( mfcc / fbank / mel / linear )', required=False)
    parser.add_argument('--delta', default=False, type=boolean_string, help='Append Delta', required=False)
    parser.add_argument('--delta_delta', default=False, type=boolean_string, help='Append Delta Delta', required=False)
    parser.add_argument('--apply_cmvn', default=False, type=boolean_string, help='Apply CMVN on feature',
                        required=False)
    parser.add_argument('--n_jobs', default=16, type=int, help='Number of jobs used for feature extraction',
                        required=False)
    parser.add_argument('--name', default='None', type=str, help='Name of the output directory', required=False)

    args = parser.parse_args()

    mel_dim = 80 * (1 + int(args.delta) + int(args.delta_delta))
    dim = mel_dim

    # Acoustic Feature Extraction & Make Data Table
    acoustic_preprocess(args, dim, args.name)


if __name__ == '__main__':
    # main()
    generate_feat_file('/home/mlxu/dataall/Aishell_npy',
                       '/home/mlxu/PycharmProjects/neural_sp/data',
                       type='eval_80')
    # generate_transcript()

    
