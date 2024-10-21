######################################################
## sound event detection model
## music ringbacktone vs voice
######################################################

import numpy as np
import pickle
from collections import Counter
from scipy.spatial.distance import cdist
import webrtcvad
from collections import Counter
vad = webrtcvad.Vad(3)
def tensor_to_bytes(audio_tensor):
    audio_np = audio_tensor.squeeze().numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    return audio_int16.tobytes()

def is_speech_chunk(audio_chunk, sample_rate=16000):
    frame_duration = 30 
    frame_length = int(sample_rate * frame_duration / 1000)
    audio_bytes = tensor_to_bytes(audio_chunk)
    if len(audio_bytes) != frame_length * 2:  
        raise ValueError("Frame size does not match the required length for VAD.")
    return vad.is_speech(audio_bytes, sample_rate)

def is_speech(chunk, chunk_size = 16000, sample_rate = 16000):
    is_speech_list = []
    for j in range(0, chunk_size, int(sample_rate * 0.03)): 
        frame = chunk[:, j:j + int(sample_rate * 0.03)]
        if frame.shape[1] == int(sample_rate * 0.03):
            is_speech_list.append(is_speech_chunk(frame, sample_rate))
    counter = Counter(is_speech_list)
    return counter.most_common(1)[0][0]

class CLSforRing:
    def __init__(self, music_model):
        # if music_model == "DT":
        #     with open('ring_music_DT.pkl', 'rb') as f:
        #         self.music_ring = pickle.load(f)
        # elif music_model == "RF":
        #     with open('ring_music_RF.pkl', 'rb') as f:
        #         self.music_ring = pickle.load(f)
        # else:
        #     raise Warning("Select music model : DT / RF")
        self.labels = np.load('ring_label.npy')
        sys_specs = np.load('ring_spec.npy')
        self.sys_specs = np.reshape(sys_specs, (sys_specs.shape[0], -1))
        with open('idx2ring.pickle', 'rb') as fr:
            self.idx2ring = pickle.load(fr)
        self.ring_preds = []
    
    def calculate_similarity(self, spec):
        return cdist(spec.numpy(), self.sys_specs)
    

    # def sys_cls(self, tmp, topk):
    #     distances = self.calculate_similarity(tmp)
    #     top_idxs = np.argsort(distances)[0][:topk]
    #     return [self.idx2ring[self.labels[i]] for i in top_idxs]
    
    # def cls(self, x, topk = 1):
    #     is_sys = bool(self.music_ring.predict(x).item())
    #     if is_sys:
    #         self.ring_preds += self.sys_cls(x, topk = topk)
    #         return self.sys_cls(x, topk = topk)
    #     else:
    #         self.ring_preds += ["no sys"]*topk
    #         return "no sys"
    
    def cls(self, x):
        distances = self.calculate_similarity(x)
        if np.sort(distances)[0][0] < 0.05:
            top_idxs = np.argsort(distances)[0][0]
            return self.idx2ring[self.labels[top_idxs]]
        else:
            return "no sys"
            
    def return_out(self):
        # if ("voice" in self.ring_preds):
        #     return ['no sys']
        counter = Counter(self.ring_preds)
        # return counter.most_common(1)[0][0]
        most_common_count = counter.most_common(1)[0][1]
        return [element for element, count in counter.items() if count == most_common_count]