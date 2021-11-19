# `vad remove silence`

This will remove the start&end silence from your audio(.wav file)

original source: [video-remove-silence](https://github.com/excitoon/video-remove-silence)
Enhancing by adding VAD(Voice Activity Detection) to video remove silence.

<pre>
<code>
mkdir files
python vremove_silence_vad.py
</code>
</pre>

### What I used from original source
* the original source detect all silence in Video/Audio by dB
* Changed source only for Audio

### What I changed & enhanced
* Using VAD
* Sound normalization
* Remove start/end silence on .wav file.

### Dependencies

- Python >= 3.5
- scipy
- webrtcvad
- pydub
- numpy <= 1.19.5
