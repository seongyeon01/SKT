import os
import pickle
ring_path = "./data/ring" # path for system ringbacktone
ring_list = [f"{ring_path}/{w}" for w in os.listdir(ring_path) if not (w.startswith('silence500ms_wb'))] # 1초 이하 제거
ring2idx = {r:i for i, r in enumerate(ring_list)}
idx2ring = {i:os.path.basename(r) for i, r in enumerate(ring_list)}

ring_specs = []
labels = []
for r in (ring_list):
    ring = load_and_resample(r, target_sr)
    for s in range(0, ring.shape[1]-target_sr, target_sr):
        ring_sec = extract_sec(ring, target_sr, start = int(s/target_sr))
        ring_filter = preprocess(ring_sec)
        ring_specs.append(ring_filter.squeeze())
        labels.append(ring2idx[r])

np.save('ring_label.npy', np.array(labels))
np.save('ring_spec.npy', torch.stack(ring_specs).numpy())
with open('idx2ring.pickle','wb') as fw:
    pickle.dump(idx2ring, fw)