#!/usr/bin/env python
# coding: utf-8

from scipy.io import wavfile
import wave
import webrtcvad
import numpy as np
import glob
import os
import wave
from pydub import AudioSegment
import collections
import argparse

def find_silences(filename):
    global args
    blend_duration = 0.005
    with wave.open(filename) as wav:
        size = wav.getnframes()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frame_rate = wav.getframerate()
        max_value = 1 << (8 * sample_width - 1)
        half_blend_frames = int(blend_duration * frame_rate / 2)
        blend_frames = half_blend_frames * 2
        assert size > blend_frames > 0
        square_threshold = max_value ** 2 * 10 ** (args.threshold_level / 10)
        blend_squares = collections.deque()
        blend = 0

        def get_values():
            frames_read = 0
            while frames_read < size:
                frames = wav.readframes(min(0x1000, size - frames_read))
                frames_count = len(frames) // sample_width // channels
                for frame_index in range(frames_count):
                    yield frames[frame_index*channels*sample_width:(frame_index+1)*channels*sample_width]
                frames_read += frames_count

        def get_is_silence(blend):
            results = 0
            frames = get_values()
            for index in range(half_blend_frames):
                frame = next(frames)
                square = 0
                for channel in range(channels):
                    value = int.from_bytes(frame[sample_width*channel:sample_width*channel+sample_width], 'little', signed=True)
                    square += value*value
                blend_squares.append(square)
                blend += square
            for index in range(size-half_blend_frames):
                frame = next(frames)
                square = 0
                for channel in range(channels):
                    value = int.from_bytes(frame[sample_width*channel:sample_width*channel+sample_width], 'little', signed=True)
                    square += value*value
                blend_squares.append(square)
                blend += square
                if index < half_blend_frames:
                    yield blend < square_threshold * channels * (half_blend_frames + index + 1)
                else:
                    result = blend < square_threshold * channels * (blend_frames + 1)
                    if result:
                        results += 1
                    yield result
                    blend -= blend_squares.popleft()
            for index in range(half_blend_frames):
                blend -= blend_squares.popleft()
                yield blend < square_threshold * channels * (blend_frames - index)

        is_silence = get_is_silence(blend)

        def to_regions(iterable):
            iterator = enumerate(iterable)
            while True:
                try:
                    index, value = next(iterator)
                except StopIteration:
                    return
                if value:
                    start = index
                    while True:
                        try:
                            index, value = next(iterator)
                            if not value:
                                yield start, index
                                break
                        except StopIteration:
                            yield start, index+1
                            return

        threshold_frames = int(args.threshold_duration * frame_rate)
        silence_regions = ( (start, end) for start, end in to_regions(is_silence) if end-start >= blend_duration )
        silence_regions = ( (start + (half_blend_frames if start > 0 else 0), end - (half_blend_frames if end < size else 0)) for start, end in silence_regions )
        silence_regions = [ (start, end) for start, end in silence_regions if end-start >= threshold_frames ]
        including_end = len(silence_regions) == 0 or silence_regions[-1][1] == size
        silence_regions = [ (start/frame_rate, end/frame_rate) for start, end in silence_regions ]
        # print(args.save_silence)
        if args.save_silence:
            with wave.open(args.save_silence, 'wb') as out_wav:
                out_wav.setnchannels(channels)
                out_wav.setsampwidth(sample_width)
                out_wav.setframerate(frame_rate)
                for start, end in silence_regions:
                    wav.setpos(start)
                    frames = wav.readframes(end-start)
                    out_wav.writeframes(frames)

    return silence_regions, including_end

def transform_duration(duration):
    global args
    return args.constant + args.sublinear * math.log(duration + 1) + args.linear * duration

def format_offset(offset):
    return '{}:{}:{}'.format(int(offset) // 3600, int(offset) % 3600 // 60, offset % 60)

def closest_frames(duration, frame_rate):
    return int((duration + 1 / frame_rate / 2) // (1 / frame_rate))

def compress_audio(wav, start_frame, end_frame, result_frames):
    # print(start_frame, end_frame, result_frames)
    if result_frames == 0:
        return b''
    elif result_frames == end_frame - start_frame:
        # print("same")
        wav.setpos(start_frame)
        return wav.readframes(result_frames)
    else:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frame_width = sample_width*channels
        if result_frames*2 <= end_frame - start_frame:
            left_length = result_frames
            right_length = result_frames
        else:
            left_length = (end_frame - start_frame + 1) // 2
            right_length = end_frame - start_frame - left_length
        crossfade_length = right_length + left_length - result_frames
        crossfade_start = (result_frames - crossfade_length) // 2
        wav.setpos(start_frame)
        left_frames = wav.readframes(left_length)
        wav.setpos(end_frame - right_length)
        right_frames = wav.readframes(right_length)
        result = bytearray(b'\x00'*result_frames*frame_width)
        result[:(left_length-crossfade_length)*frame_width] = left_frames[:-crossfade_length*frame_width]
        result[-(right_length-crossfade_length)*frame_width:] = right_frames[crossfade_length*frame_width:]
        for i in range(crossfade_length):
            r = i / (crossfade_length - 1)
            l = 1 - r
            for channel in range(channels):
                signal_left = int.from_bytes(left_frames[(left_length-crossfade_length+i)*frame_width+channel*sample_width:(left_length-crossfade_length+i)*frame_width+(channel+1)*sample_width], 'little', signed=True)
                signal_right = int.from_bytes(right_frames[i*frame_width+channel*sample_width:i*frame_width+(channel+1)*sample_width], 'little', signed=True)
                result[(left_length-crossfade_length+i)*frame_width+channel*sample_width:(left_length-crossfade_length+i)*frame_width+(channel+1)*sample_width] = int(signal_left*l + signal_right*r).to_bytes(sample_width, 'little', signed=True)
        return result

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    frames = []
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        frames.append(Frame(audio[offset:offset + n], timestamp, duration))
        timestamp += duration
        offset += n
    
    return frames


parser = argparse.ArgumentParser()
parser.add_argument('-path', type=str, default='files/', help='path to video')
parser.add_argument('-threshold-duration', type=float, default=0.2, help='threshold duration in seconds')
parser.add_argument('-check', type=bool, default=True, help='path to text file')
parser.add_argument('-p', type=str, default='results/', help='path to video')
parser.add_argument('--threshold-level', type=float, default=-35, help='threshold level in dB')
parser.add_argument('--constant', type=float, default=0, help='duration constant transform value')
parser.add_argument('--sublinear', type=float, default=0, help='duration sublinear transform factor')
parser.add_argument('--linear', type=float, default=0.1, help='duration linear transform factor')
parser.add_argument('--save-silence', type=str, help='filename for saving silence')
parser.add_argument('--recalculate-time-in-description', type=str, help='path to text file')
parser.add_argument('-f')
args = parser.parse_args()


# from scipy.io.wavfile import write



if __name__ == '__main__':
    
    dir_path = args.path +'*.wav'
    paths = glob.glob(dir_path)
    file_num = len(paths)
    vad = webrtcvad.Vad()
    vad.set_mode(3)
    
    print('Num of file: ', file_num)
    cnt = 0
    for path in paths:
        cnt += 1
        print(cnt, '/', file_num, path)
        sample_rate, samples = wavfile.read(path)
        # print('sample rate : {}, samples.shape : {}'.format(sample_rate, samples.shape))

        
        # 10, 20, or 30
        frame_duration = 10 # ms
        frames = frame_generator(frame_duration, samples, sample_rate)
        flag = True
        for i, frame in enumerate(frames):
            if vad.is_speech(frame.bytes, sample_rate):
                if flag:
                    start_idx = i
                    flag = False
                else:
                    end_idx = i
        if start_idx > 1:
            start_idx -= 1
        if end_idx < len(frames):
            end_idx += 1

        audio_start_frame = int(start_idx/100.0*sample_rate*2)
        audio_end_frame = int(end_idx/100.0*sample_rate*2)
        audio_result_frames = audio_end_frame - audio_start_frame

        dst = args.p + path.split(args.path)[-1]
        if not os.path.isdir(args.p):
            os.mkdir(args.p)

        wav = wave.open(path, mode='rb')
        out_wav = wave.open(dst, mode='wb')
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        audio_frame_rate = wav.getframerate()
        out_wav.setnchannels(channels)
        out_wav.setsampwidth(sample_width)
        out_wav.setframerate(audio_frame_rate)

        out_wav.writeframes(compress_audio(wav, audio_start_frame, audio_end_frame, audio_result_frames))
        # write(dst, sample_rate, samples_cut)

        out_wav.close()

        silences, including_end = find_silences(dst)
        if silences[0][0] == 0.0:
            start_gap = silences[0][1] - silences[0][0]
        else:
            start_gap = 0
        if including_end:
            end_gap = silences[-1][1] - silences[-1][0]
        else:
            end_gap = 0

        seg = AudioSegment.silent(duration=200)
        song = AudioSegment.from_wav(dst)

        if start_gap < 0.2:
            start_gap += 0.2
            song = seg + song
        if end_gap < 0.2:
            end_gap += 0.2
            song = song + seg
        song.export(dst, format="wav")
    
        print("start/end gap: ", round(start_gap, 2), round(end_gap, 2))
        print()

# import librosa
# sample_rate, samples = wavfile.read('EN_F_2_R_S24-2_017.wav')
# print(sample_rate)
# y, sr = librosa.load('EN_F_2_R_S24-2_017.wav', sr=sample_rate)
# resample = librosa.resample(y, sr, 16000)

# ipd.Audio(samples, rate=sample_rate)
