from numpy.core.defchararray import add
from Utils import *
import os
import glob

stft_para = {
    'stftn': 3840,
    'fStart': [1, 4, 8, 14, 31],
    'fEnd': [3, 7, 13, 30, 50],
    'fs': 128,
    'window': 30,
}

data_dir = '../SS3/data/'
label_dir = '../SS3/label/'

output_dir = '../data/'
output_dir_psd = os.path.join(output_dir, 'PSD')
output_dir_eog = os.path.join(output_dir, 'EOG')
output_dir_label = os.path.join(output_dir, 'label')

def GenerateTopographic(data, sub_id):
    '''
    data: NumPy Tensor (Sample, EEG Channels, Time Length)
    sub_id: int
    '''
    data = butter_bandpass_filter(data, 0.5, 50, 128, 4)
    print("Temporal data shape:", data.shape)  # Data shape (1005, 20, 7680)

    psd_frequency = []
    for sample in range(data.shape[0]):
        psd, _ = DE_PSD(data[sample], stft_para)  # PSD shape (20, 5)
        psd_frequency.append(psd)
    psd_frequency = np.array(psd_frequency)
    print("PSD data shape:", psd_frequency.shape)
    MY_frequency = norm(psd_frequency)

    heatmap = convert_heat(MY_frequency)

    heatmap_spectral = np.zeros(
        [MY_frequency.shape[0], 5, 16, 16], dtype='float32')
    for ep in range(heatmap.shape[0]):
        for hz in range(heatmap.shape[1]):
            heatmap_spectral[ep, hz, :, :] = grid_data(heatmap[ep][hz])
    heatmap_spectral = heatmap_spectral[:, :, :, :, np.newaxis]
    return heatmap_spectral

def main():
    if os.path.exists(output_dir) is not True:
        os.mkdir(output_dir)
    if os.path.exists(output_dir_psd) is not True:
        os.mkdir(output_dir_psd)
    if os.path.exists(output_dir_eog) is not True:
        os.mkdir(output_dir_eog)
    if os.path.exists(output_dir_label) is not True:
        os.mkdir(output_dir_label)

    for sub_id in range(1, 65):
        print(f'Subject {sub_id}')
        if sub_id in (43, 49): # Exclude 43 and 49
            continue

        
        data = np.load(os.path.join(data_dir, f'01-03-00{sub_id:02}-Data.npy'))
        # data's shape: [sample, channel, length] where
        #       CHANNEL: 0 -> ECG channel, 1-20 -> EEG channels, 
        #                21-23 -> EMG channels, 24->25 EOG channels
        label = np.load(os.path.join(label_dir, f'subject{sub_id}.npy'))


        EEG = data[:, 1:21, :]  # Select EEG channels
        EEGRepresentation = GenerateTopographic(EEG, sub_id)
        EOGRepresentation = data[:, 24:26, :]

        EEGRepresentation = AddContext(EEGRepresentation, add_context=False).astype(np.float32)
        EOGRepresentation = AddContext(EOGRepresentation, add_context=True).astype(np.float32)
        label = AddContext(label, add_context=False)

        np.save(os.path.join(output_dir_psd, f'subject{sub_id}'), EEGRepresentation)
        np.save(os.path.join(output_dir_eog, f'subject{sub_id}'), EOGRepresentation)
        np.save(os.path.join(output_dir_label, f'subject{sub_id}'), label)

if __name__ == '__main__':
    main()
