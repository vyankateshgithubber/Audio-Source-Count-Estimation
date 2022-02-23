import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import librosa
import tensorflow as tf

model = tf.keras.models.load_model('TrimmedSTFT.h5')

app = Flask(__name__)

def freq_thresh_index(fs,n_fft,thresh=4000):
  x=librosa.fft_frequencies(fs, n_fft=512)
  print(x.shape)
  for i in range(len(x)):
    if(x[i]>thresh):
      N=i-1
      break
  return N

n=freq_thresh_index(fs=16000,n_fft=512,thresh=4000)

## Short Time Fourier Transform function
def stft(y,sr):
  S = abs(librosa.stft(y,n_fft=512,hop_length=256))
  R1= np.array(S)
  r = R1[0:n+1, :]
  return r


def prediction(test_filepath,offset=0,duration=1):
  signal, sr = librosa.load(test_filepath,sr=16000,duration=duration,offset=offset) ## offset 
  feature = stft(signal,sr)
  size = feature.shape
  feature = feature.reshape(1,size[0],size[1])
  output = model.predict(feature)
  output = np.argmax(output)
  return output


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    file=request.files['file']
    output = []
    if file.filename != '':
        file.save(file.filename)
        test_filepath = file.filename
        signal, sr = librosa.load(test_filepath,sr=16000)
        for i in range(len(signal)//sr):
            output.append(prediction(test_filepath,i,1))

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)