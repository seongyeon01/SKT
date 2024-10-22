from flask import Flask, request, jsonify, render_template
import torch
import torchaudio
import ring_voice
import os
from CLS_ring import CLSforRing
from utils import extract_sec, preprocess, forpred
from prediction_denoise import prediction

app = Flask(__name__)

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

# wav 파일을 로드하고 리샘플링하는 함수
def load_wav_and_resample(wav_path, target_sr):
    waveform, sample_rate = torchaudio.load(wav_path)
    
    # 원본 샘플 레이트와 대상 샘플 레이트가 다른 경우 리샘플링
    if sample_rate != target_sr:
        resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resample_transform(waveform)
    return waveform

def process_wav_file(wav_path):
    output_log = []  # 로그를 저장할 리스트
    output_log.append(f"<p>Processing: {wav_path}")
    sample = load_wav_and_resample(wav_path, target_sr)  # wav 파일 로드 및 리샘플링
    RING = CLSforRing('RF')
    input_filename = os.path.splitext(os.path.basename(wav_path))[0]  # 입력 파일 이름 추출
    prev_speech = False
    speech_start = None  # speech 감지 시작 시간
    music_printed = False
    sys_printed = False

    for s in range(0, sample.shape[-1] - target_sr, target_sr):
        sample_sec = extract_sec(sample, target_sr, start=int(s / target_sr))  # 샘플 추출
        sample_filter = preprocess(sample_sec, n_fft, hop_length, hann_window, target_sr, config['freq_range']).float()
        is_sys = RING.cls(sample_filter.reshape(1, -1))

        if is_sys == 'no sys':
            with torch.no_grad():
                out_dict = model(forpred(sample_filter.unsqueeze(0)).unsqueeze(0).type(torch.FloatTensor).transpose(2, 3).to(device))
            test_pred = out_dict['event_logit']
            no_speech = torch.argmax(test_pred, dim=-1).item()

            if no_speech:
                print(f"{int(s / target_sr)} : {int(s / target_sr) + 1} - music")
                if no_speech and not music_printed:
                    output_log.append("<p>music")
                    print("add music")
                    music_printed = True    
                prev_speech = False
                speech_start = None  # 음악이 감지되면 초기화
            else:
                if prev_speech:  # 두 번째 연속된 speech 감지
                    print(f"{int(s / target_sr)} : {int(s / target_sr) + 1} - speech")
                    output_log.append(f"<p>speech start {int(s / target_sr)}")
                    if speech_start is None:
                        speech_start = s  # 첫 번째 speech 시작 시간(샘플 단위)
                    
                    # 첫 speech 감지 시간부터 잘라냄
                    truncated_sample = sample[:, speech_start:]

                    #######################
                    ### Enhancement 및 저장
                    #######################
                    audio_dir_prediction = '/home/ubuntu/jupyter/SKT/submit/test/original'
                    output_wav_path = f'/home/ubuntu/jupyter/SKT/submit/test/original/{input_filename}_speech_from_{int(speech_start / target_sr)}.wav'
                    torchaudio.save(output_wav_path, truncated_sample, target_sr)
                    output_log.append(f"<p>original : {output_wav_path}")

                    dir_save_prediction = '/home/ubuntu/jupyter/SKT/submit/test/sample_denoise/'
                    audio_input_prediction = [output_wav_path]
                    audio_output_prediction = f'{input_filename}_denoise_from_{int(speech_start / target_sr)}.wav'

                    # Prediction 및 enhancement 단계
                    prediction(weights_path, name_model, audio_dir_prediction, dir_save_prediction, audio_input_prediction,
                               audio_output_prediction, 8000, 1.0, 8064, 8064, 255, 63)
                    break
                else:
                    prev_speech = True  # 첫 번째 speech 감지
        else:
            print(f"{int(s / target_sr)} : {int(s / target_sr) + 1} - {is_sys}")
            if not sys_printed:
                output_log.append(f"<p>{is_sys}")
                print(f"add {is_sys}")
            prev_speech = False
            speech_start = None  # 시스템 음성이 감지되면 초기화
    
    # 로그를 하나의 문자열로 묶어서 반환
    return ''.join(output_log)

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        # 업로드된 파일을 임시 경로에 저장
        file_path = f'/tmp/{file.filename}'
        file.save(file_path)

        # process_wav_file 함수 호출하여 처리 및 로그 반환
        log_output = process_wav_file(file_path)

        # HTML로 로그 출력
        return f"<html><body>{log_output}</body></html>"
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
