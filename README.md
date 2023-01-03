# randomCNN-voice-transfer
# Alireza Ahmadi 40114140111015
# digital signal processing
# Voice style transfer with random CNN
Maybe the fastest voice style transfer with reasonable result ?
## What is voice style transfer?
Inspired by the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) , the idea of `Neural Voice Transfer` aims at  "using Obama's voice to sing songs of Beyoncé" or something related.

We aim to:


## Highlight of my work
* Use **2-D CONV** rather than 1-D for audio spectrogram.
* Compute **grams over time-axis**.
* **Training fast**. 5-10 minutes to train and transfer on 1 single GPU(Tesla P40).
* **Do not need dataset!** You can transfer any 2 pieces of audio.(But some format of audio may occur error, then you should `sudo apt-get install libav-tools`)

## Results
**You can listen to my current result  now !** It's on soundcloud, [link1](https://soundcloud.com/mazzzystar/sets/stairway2nightcall), [link2](https://soundcloud.com/mazzzystar/sets/speech-conversion-sample).

The generated spectrogram compared with `content` and `style`.
![](picture/gen.png)

Compare the spectrogram of `gen` with `content` and `style`(X axis represents `Time Domain`, Y axis represents `Frequency Domain`),  we can find that:
* The structure is almost the same as `content`, and the **gap along frequency axis**, which determines the `voice texture` to a great extent, is more alike to the style.
* The base skeleton is **shifted upward a little bit** for being similar to the style(The style is girl's voice, which has higher frequency than boy's).

## Reproduce it yourself
```
pip install -r requirements.txt 
# remove `CUDA_VISIBLE_DEVICES` when use CPU, though it will be slow. 
CUDA_VISIBLE_DEVICES=0 python train.py -content input/boy18.wav -style input/girl52.wav
```
Tips: change `3x1` CONV to `3x3` CONV can get smoother generated spectrogram.

### But..does the `gram` of random CNN output really works ?
Below is my experiments result of using `texture gram`  after 1-layer RandomCNN  to capture speaker identity by putting them as **the only feature** in a simple nearest neighbor speaker identification system. The table shows the result of speaker identification accuracy of this system over the first 15 utterances of 30 first speakers of the VCTK dataset, along with 100 utterances of 4 first speakers.

| Speakers        | Train/Test           | Accuracy  |
| ------------- |:-------------:| -----:|
| 30     | 270/180 | 45.6%|
| 4      | 240/160      |   92.5% |

It seems `texture gram along time-axis` really captured something, you can check it by:
```
python vctk_identify
```

# main source code = https://github.com/mazzzystar/randomCNN-voice-transfer/blob/master/README.md
# my acount in github = https://github.com/alirezaahmadiii    
# linkedin = https://www.linkedin.com/in/alireza-ahmadi-245214258

# videos that are going to help you to learn more about this project and more about colab and matrix and machine learning :
https://drive.google.com/drive/folders/1Sd_onkfdwq63tsg20tfum7vBUTCaQa-K?usp=sharing .1
https://colab.research.google.com/drive/10-8X59ey1gYBU2Uj3s-2fco1cLcKwGul?usp=sharing .2
 https://drive.google.com/drive/folders/1Sd_onkfdwq63tsg20tfum7vBUTCaQa-K?usp=sharing .3
https://drive.google.com/drive/folders/1o5bZ28hl75eHwlEf-SfGKch6km8tYTZG .4
https://www.aparat.com/v/wPWKh .5
https://www.aparat.com/v/FCtZ4 .6
https://aparat.com/v/rncBI .7
https://aparat.com/v/CzVJn .8
https://aparat.com/v/OZSFB .9
https://www.aparat.com/v/wPWKh .10
https://www.aparat.com/v/FCtZ4 .11
https://www.aparat.com/v/2o9Hb .12
https://www.aparat.com/v/jbf2z .13
https://drive.google.com/drive/folders/1y8bNyDQwvbbm60ih1fMeuu1Apnv76zKC?usp=share_li .14
nk
https://drive.google.com/drive/folders/1Sd_onkfdwq63tsg20tfum7vBUTCaQa- .15
K?usp=share_link
https://drive.google.com/drive/folders/1o5bZ28hl75eHwlEf-SfGKch6km8tYTZG?usp=share_link .16
