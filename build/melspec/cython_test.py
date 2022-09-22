import numpy as np

from build.melspec import melspec
import torchaudio

def main():
    wav_path = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess\iphone11_0908\wave_split\1662604080865-000.wav'
    torch_wav, sr = torchaudio.load(wav_path, normalize=False)
    np_wav = torch_wav.detach().cpu().numpy()
    print(torch_wav, np_wav)
    melspec_process = melspec.PyMelspec(16000, 12, 25, 10, 40, 50, 8000)
    m = melspec_process.WavformtoMelspec(np_wav[0])
    m_reshape = np.reshape(np.array(m), [128, 59])
    print(m_reshape)

if __name__ == '__main__':
    main()