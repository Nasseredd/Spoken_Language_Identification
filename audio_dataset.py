########################### NOTES ##########################
# Monday 10.01.2022 
    # 21:53 DONE: create AudioDataset that gets mel spectrograms as features and the language as label
############################################################
import glob 
import librosa
import torch 
from torch.utils.data import Dataset, DataLoader



class AudioDataset(Dataset):
    def __init__(self, audios_directory):
        '''initialization of the attributes'''
        #self.sample_rate = sample_rate
        self.audios_directory = audios_directory
        self.audio_paths = self.get_audio_paths_list(audios_directory, extension='*.wav')
        print(self.audio_paths)
        self.language2index = {'en':0, 'de':1, 'es':2}
    
    def __len__(self):
        '''return the number of audio files in the audio_dir'''
        return len(self.audio_paths)
    
    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        print(audio_path)
        mel_spectrogram_tensor = self.mel_spectrogram(audio_path)
        audio_name = audio_path.split('/')[1]
        language = audio_name[:2] # Label : the 2 first characters of the audio_path (en, de, es)
        language_index = self.language2index[language] # encode the Label 
        language_tensor = torch.Tensor([language_index])
        return mel_spectrogram_tensor, language_tensor
    
    # Mel Spectrogram  
    def mel_spectrogram(self, audio_path):
        '''return the tensor of the mel spectrogram of the audio in shape (features, frames)'''
        y, sr = librosa.load(audio_path)  # y: audio time-series, sr: sample rate
        print(y.shape)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr) 
        mel_spectrogram = torch.from_numpy(mel_spectrogram)
        return mel_spectrogram

    # Getters 
    def get_audio_paths_list(self, directory, extension='*.wav'):
        '''returns the list of the audio_paths'''
        return [audio_path for audio_path in glob.iglob(directory + '/' + extension)]
            

# Main 
if __name__ == '__main__':
    audios_directory = 'Audios_tmp'
    # import the dataset 
    dataset = AudioDataset(audios_directory)
    # # create the dataloader 
    # dataloader = DataLoader(dataset, shuffle=True, batch_size=8)
    # # display one single batch of a loader
    # for batch_size, (mel_spectrogram, language) in enumerate(dataloader):
    #     print(mel_spectrogram.shape)
    #     print(language)
    #     break
    print(dataset[0])
