########################### NOTES ##########################
# Monday 10.01.2022 
    # 21:53 DONE: create AudioDataset that gets mel spectrograms as features and the language as label
############################################################
import glob 
import librosa
from torch.utils.data import Dataset 



class AudioDataset(Dataset):
    def __init__(self, audios_directory):
        '''initialization of the attributes'''
        #self.sample_rate = sample_rate
        self.audios_directory = audios_directory
        self.audio_paths = self.get_audio_paths_list(audios_directory, extension='*.wav')
        self.language2index = {'en':0, 'de':1, 'es':2}
    
    def __len__(self):
        '''return the number of audio files in the audio_dir'''
        return len(self.audio_paths)
    
    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        mel_spectrogram = self.mel_spectrogram(audio_path)
        language = audio_path[:2] # Label : the 2 first characters of the audio_path (en, de, es)
        language_index = self.language2index[language]
        return mel_spectrogram, language_index
    
    # Mel Spectrogram  
    def mel_spectrogram(audio_path):
        '''return the mel spectrogram of the audio in shape (features, frames)'''
        y, sr = librosa.load(audio_path)  # y: audio time-series, sr: sample rate
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr) 
        return mel_spectrogram

    # Getters 
    def get_audio_paths_list(self, directory, extension='*.wav'):
        '''returns the list of the audio_paths'''
        return [audio_path for audio_path in glob.iglob(directory + extension)]
            

# Main 
if __name__ == '__main__':
    audios_directory = 'Audios_tmp'
    dataset = AudioDataset(audios_directory)
