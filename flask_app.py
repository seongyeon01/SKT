from flask import Flask, request, jsonify, render_template, Response
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

# 실시간 스트리밍을 위한 generator 함수
def process_wav_file(wav_path):
    yield f"<p>Processing: {wav_path}</p>"
    sample = load_wav_and_resample(wav_path, target_sr)  # wav 파일 로드 및 리샘플링
    RING = CLSforRing('RF')
    input_filename = os.path.splitext(os.path.basename(wav_path))[0]  # 입력 파일 이름 추출
    prev_speech = False
    speech_start = None  # speech 감지 시작 시간
    music_printed = False
    last_is_sys = None  # 마지막으로 감지된 is_sys 상태를 저장
    is_sys_printed = False  # is_sys가 한 번 출력되었는지 여부를 추적하는 플래그 변수

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
                if not music_printed:
                    yield "<p>music</p>"
                    music_printed = True    
                prev_speech = False
                speech_start = None  # 음악이 감지되면 초기화
            else:
                if prev_speech:  # 두 번째 연속된 speech 감지
                    yield f"<p>speech start {int(s / target_sr)}</p>"
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
                    yield f"<p>original : {output_wav_path}</p>"

                    dir_save_prediction = '/home/ubuntu/jupyter/SKT/submit/test/sample_denoise/'
                    audio_input_prediction = [output_wav_path]
                    audio_output_prediction = f'{input_filename}_denoise_from_{int(speech_start / target_sr)}.wav'

                    # Prediction 및 enhancement 단계
                    prediction(weights_path, name_model, audio_dir_prediction, dir_save_prediction, audio_input_prediction,
                               audio_output_prediction, 8000, 1.0, 8064, 8064, 255, 63)
                    yield f"<p>denoise : {audio_output_prediction}</p>"
                    break
                else:
                    prev_speech = True  # 첫 번째 speech 감지
        else:
            if not is_sys_printed:  # is_sys가 한 번도 출력되지 않은 경우에만 출력
                yield f"<p>{is_sys}</p>"
                is_sys_printed = True  # 출력되었음을 기록
            last_is_sys = is_sys  # 마지막 is_sys 값을 기록
            speech_start = None  # 시스템 음성이 감지되면 초기화

    # 마지막으로 감지된 is_sys만 로그에 추가
    if last_is_sys and not is_sys_printed:
        yield f"<p>{last_is_sys}</p>"


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

        # 실시간 로그 스트리밍
        return Response(process_wav_file(file_path), mimetype='text/html')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
