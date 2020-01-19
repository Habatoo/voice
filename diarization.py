import wave
import pickle
import contextlib

import librosa
# A python package for music and audio analysis.

import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist

import webrtcvad
# a python interface to the WebRTC Voice Activity Detector (VAD).
# A VAD classifies a piece of audio data as being voiced or unvoiced.
# It can be useful for telephony and speech recognition.

import collections
import copy
import os
from os import listdir
from IPython.display import clear_output
from sklearn.cluster import SpectralClustering
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(FILE_ANALISE, N_CLUSTERS=2):
    # Feature Extraction
    def extract_features(y, sr, window, hop, n_mfcc):
        mfcc = librosa.feature.mfcc(
                y=y, sr=sr, hop_length=int(hop*sr),
                n_fft=int(window*sr), n_mfcc=n_mfcc, dct_type=2)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        stacked = np.vstack((mfcc, mfcc_delta, mfcc_delta2))
        return stacked.T

    # code modified for compactness
    # orignal code
    # https://github.com/wiseman/py-webrtcvad/blob/master/example.py
    def write_wave(path, audio, sample_rate):
        with contextlib.closing(wave.open(path, 'wb')) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio)

    class Frame(object):
        def __init__(self, bytes, timestamp, duration):
            self.bytes = bytes
            self.timestamp = timestamp
            self.duration = duration

    def frame_generator(frame_duration_ms, audio, sample_rate):
        n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(audio):
            yield Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def vad_collector(
            sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
        num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        voiced_frames = []
        for frame in frames:
            is_speech = vad.is_speech(frame.bytes, sample_rate)
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
            else:
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len(
                        [f for f, speech in ring_buffer if not speech])
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    triggered = False
                    yield b''.join([f.bytes for f in voiced_frames])
                    ring_buffer.clear()
                    voiced_frames = []
        if voiced_frames:
            yield b''.join([f.bytes for f in voiced_frames])

    def map_adaptation(
            gmm, data, max_iterations=300,
            likelihood_threshold=1e-20, relevance_factor=16):
        N = data.shape[0]
        D = data.shape[1]
        K = gmm.n_components

        mu_new = np.zeros((K, D))
        n_k = np.zeros((K, 1))

        mu_k = gmm.means_
        cov_k = gmm.covariances_
        pi_k = gmm.weights_

        old_likelihood = gmm.score(data)
        new_likelihood = 0
        iterations = 0
        while(abs(
                old_likelihood - new_likelihood) >
                likelihood_threshold and iterations < max_iterations):
            iterations += 1
            old_likelihood = new_likelihood
            z_n_k = gmm.predict_proba(data)
            n_k = np.sum(z_n_k, axis=0)

            for i in range(K):
                temp = np.zeros((1, D))
                for n in range(N):
                    temp += z_n_k[n][i]*data[n, :]
                mu_new[i] = (1/n_k[i])*temp

            adaptation_coefficient = n_k/(n_k + relevance_factor)
            for k in range(K):
                mu_k[k] = (
                        adaptation_coefficient[k] * mu_new[k]) + (
                                (1 - adaptation_coefficient[k]) * mu_k[k])
            gmm.means_ = mu_k

            log_likelihood = gmm.score(data)
            new_likelihood = log_likelihood
            print(log_likelihood)
        return gmm

    # Setings
    SR = 16000  # sample rate
    N_MFCC = 13  # number of MFCC to extract
    N_FFT = 0.032  # length of the FFT window in seconds
    HOP_LENGTH = 0.010  # number of samples between successive frames in sec

    N_COMPONENTS = 16  # number of gaussians
    COVARINACE_TYPE = 'full'  # cov type for GMM

    y = []
    # LOAD_SIGNAL = False
    LOAD_SIGNAL = True
    if LOAD_SIGNAL:
        y, sr = librosa.load(FILE_ANALISE, sr=SR)
        pre_emphasis = 0.97
        y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # MAKE_CHUNKS = False
    MAKE_CHUNKS = True

    if MAKE_CHUNKS:
        vad = webrtcvad.Vad(2)
        audio = np.int16(y/np.max(np.abs(y)) * 32768)

        frames = frame_generator(10, audio, sr)
        frames = list(frames)
        segments = vad_collector(sr, 50, 200, vad, frames)

        if not os.path.exists('data/chunks'):
            os.makedirs('data/chunks')

        for i, segment in enumerate(segments):
            chunk_name = 'data/chunks/chunk-%003d.wav' % (i,)
            write_wave(
                    chunk_name, segment[0: len(segment)-int(100*sr/1000)], sr)

    # extract MFCC, first and second derivatives
    FEATURES_FROM_FILE = True
    # FEATURES_FROM_FILE = False

    feature_file_name = 'data/param/features_{0}.pkl'.format(N_MFCC)

    if FEATURES_FROM_FILE:
        ubm_features = pickle.load(open(feature_file_name, 'rb'))
    else:
        ubm_features = extract_features(
                np.array(y), sr, window=N_FFT, hop=HOP_LENGTH, n_mfcc=N_MFCC)
        ubm_features = preprocessing.scale(ubm_features)
        pickle.dump(ubm_features, open(feature_file_name, "wb"))

    # UBM Train
    UBM_FROM_FILE = True
    # UBM_FROM_FILE = False

    ubm_file_name = 'data/param/ubm_{0}_{1}_{2}MFCC.pkl'.format(
            N_COMPONENTS, COVARINACE_TYPE, N_MFCC)

    if UBM_FROM_FILE:
        ubm = pickle.load(open(ubm_file_name, 'rb'))
    else:
        ubm = GaussianMixture(
                n_components=N_COMPONENTS, covariance_type=COVARINACE_TYPE)
        ubm.fit(ubm_features)
        pickle.dump(ubm, open(ubm_file_name, "wb"))
    # print(ubm.score(ubm_features))

    SV = []
    num_chunk = len(listdir(os.getcwd()+'\data\chunks'))
    for i in range(num_chunk):
        clear_output(wait=True)
        fname = 'data/chunks/chunk-%003d.wav' % (i,)
        # print('UBM MAP adaptation for {0}'.format(fname))
        y_, sr_ = librosa.load(fname, sr=None)
        f_ = extract_features(
                y_, sr_, window=N_FFT, hop=HOP_LENGTH, n_mfcc=N_MFCC)
        f_ = preprocessing.scale(f_)
        gmm = copy.deepcopy(ubm)
        gmm = map_adaptation(gmm, f_, max_iterations=1, relevance_factor=16)
        sv = gmm.means_.flatten()
        try:
            sv = preprocessing.scale(sv)
        except:
            pass
        SV.append(sv)

    SV = np.array(SV)
    clear_output()
    # print(SV.shape)

    def rearrange(labels, n):
        seen = set()
        distinct = [x for x in labels if x not in seen and not seen.add(x)]
        correct = [i for i in range(n)]
        dict_ = dict(zip(distinct, correct))
        return [x if x not in dict_ else dict_[x] for x in labels]

    sc = SpectralClustering(n_clusters=N_CLUSTERS, affinity='cosine')
    labels = sc.fit_predict(SV)
    labels = rearrange(labels, N_CLUSTERS)
    print('Обработка завершена.')
    return labels


if __name__ == '__main__':
    main('data/call_center/1.wav')
