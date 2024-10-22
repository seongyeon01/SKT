from CLS_ring import CLSforRing
from utils import load_and_resample, extract_sec, preprocess, forpred
import torch
import ring_voice
from prediction_denoise import prediction
import torchaudio

device = torch.device('cuda:1')
model = ring_voice.model
model.load_state_dict(torch.load("/home/ubuntu/jupyter/SKT/log/ring_voice_binary_2/model.pt"))
model.to(device)
model.eval()

# 코덱별 샘플링 레이트 및 주파수 범위 설정
codec_config = {
    'EVS': {'sample_rate': 16000, 'freq_range': (0, 8000)},
    'AMR-WB': {'sample_rate': 16000, 'freq_range': (0, 8000)},
    'AMR': {'sample_rate': 8000, 'freq_range': (0, 4000)},
}
selected_codec = 'EVS'
config = codec_config[selected_codec]
target_sr = config['sample_rate']
n_fft = 1024  # FFT 크기
hop_length = n_fft // 4  # 일반적으로 hop_length는 n_fft의 1/4로 설정
hann_window = torch.hann_window(n_fft)

weights_path = '/home/ubuntu/jupyter/SKT/submit/weights'
name_model = 'model_unet'

import os
# import numpy as np
import librosa

test_sample1_path = "/home/ubuntu/jupyter/SKT/data/Scenario/music.wav"
test_sample2_path = "/home/ubuntu/jupyter/SKT/data/Scenario/system_o.wav" # PCM_30_wb
test_sample3_path = "/home/ubuntu/jupyter/SKT/data/Scenario/system_x1.wav" # PCM_15_wb

test_sample4_path = "/home/ubuntu/jupyter/SKT/data/Scenario/music+speech.wav"
test_sample5_path = "/home/ubuntu/jupyter/SKT/data/Scenario/music+speech(noise).wav"

test_sample6_path = "/home/ubuntu/jupyter/SKT/data/Scenario/system_o+speech.wav" # ANNC_1508_wb + start @ 60sec
test_sample7_path = "/home/ubuntu/jupyter/SKT/data/Scenario/system_o+speech(noise).wav" # INTL_TERM_ANM_wb + start @ 4sec

for path in [test_sample1_path, test_sample2_path, test_sample3_path, test_sample4_path, test_sample5_path, test_sample6_path, test_sample7_path]:
    print("\n",path)
    sample = load_and_resample(path, target_sr)
    RING = CLSforRing('RF')
    input_filename = os.path.splitext(os.path.basename(path))[0]  # 입력 파일 이름 추출
    prev_speech = False
    speech_start = None
    
    for s in range(0, sample.shape[-1]- target_sr, target_sr):
        sample_sec = extract_sec(sample, target_sr, start = int(s/target_sr))
        sample_filter = preprocess(sample_sec, n_fft, hop_length, hann_window, target_sr, config['freq_range']).float()
        is_sys = RING.cls(sample_filter.reshape(1,-1))
        if is_sys == 'no sys':
            with torch.no_grad():
                out_dict = model(forpred(sample_filter.unsqueeze(0)).unsqueeze(0).type(torch.FloatTensor).transpose(2,3).to(device))
            test_pred=out_dict['event_logit']
            no_speech = torch.argmax(test_pred, dim = -1).item()
            if no_speech:
                print(f"{int(s/target_sr)} : {int(s/target_sr)+1} - music") 
                prev_speech = False
                speech_start = None
            else:
                if prev_speech:
                    print(f"{int(s/target_sr)} : {int(s/target_sr)+1} - speech")
                    if speech_start is None:  # 처음 speech가 감지된 시간을 기록
                        speech_start = s
                    #wav로 저장
                    
                    #######################
                    ### Enhancement
                    #######################
                    # 경로 및 파라미터들 직접 작성
                    
                    audio_dir_prediction = '/home/ubuntu/jupyter/SKT/submit/test/original'
                    truncated_sample = sample[:, speech_start:]
                    output_wav_path = f'/home/ubuntu/jupyter/SKT/submit/test/original/{input_filename}.wav'
                    torchaudio.save(output_wav_path, truncated_sample, target_sr)
                    print(f"original : {output_wav_path}")

                    dir_save_prediction = '/home/ubuntu/jupyter/SKT/submit/test/sample_denoise/'
                    audio_input_prediction = [output_wav_path]
                    audio_output_prediction = f'{input_filename}_denoise.wav'

                    # Prediction 및 enhancement 단계
                    prediction(weights_path, name_model, audio_dir_prediction, dir_save_prediction, audio_input_prediction,
                    audio_output_prediction, 8000, 1.0, 8064, 8064, 255, 63)
                    break
                else:
                    prev_speech = True
        else:
            print(f"{int(s/target_sr)} : {int(s/target_sr)+1} - {is_sys}") 
            prev_speech = False
            speech_start = None