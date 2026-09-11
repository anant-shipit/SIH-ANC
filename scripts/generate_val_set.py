import os
import random
from pathlib import Path
from sih26052.data.mixer import mix_pair, MixerConfig
from sih26052.data.manifest import ManifestWriter

clean_dir = Path(os.path.expanduser("~/Downloads/datasets/processed/clean"))
noise_dir = Path(os.path.expanduser("~/Downloads/datasets/processed/noise/broadband"))
out_dir = Path("data/val_set")
out_dir.mkdir(parents=True, exist_ok=True)

cleans = list(clean_dir.rglob("*.wav"))
noises = list(noise_dir.rglob("*.wav"))

print(f"Found {len(cleans)} clean and {len(noises)} noise files.")
if not cleans or not noises:
    print("Not enough files to generate validation set.")
    exit(1)

config = MixerConfig()
num_pairs = 100

with ManifestWriter(out_dir / "manifest_val.jsonl") as mw:
    for i in range(num_pairs):
        c_path = random.choice(cleans)
        n_path = random.choice(noises)
        
        try:
            res = mix_pair(c_path, n_path, config, "broadband")
            
            out_noisy = out_dir / f"val_{i:03d}_noisy.wav"
            out_clean = out_dir / f"val_{i:03d}_clean.wav"
            
            import soundfile as sf
            sf.write(str(out_noisy), res.noisy, 16000)
            sf.write(str(out_clean), res.clean, 16000)
            
            dur = len(res.clean) / 16000
            mw.write_entry(out_noisy, out_clean, res.snr_db, res.noise_class, "broadband", dur)
        except Exception as e:
            print(f"Skipped pair {i}: {e}")

print("Validation set generated successfully at data/val_set/manifest_val.jsonl!")
