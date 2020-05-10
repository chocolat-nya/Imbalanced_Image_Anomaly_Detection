# What's for

Anomaly detection of images. It works if NG image number is too small compared with OK image number.

# Usage

throw away .gitignore files in image/ok/ & image/ng/

put ok images in images/ok/

put ng images in images/ng/

python arcface_efficientnet.py



# Option

When you want to change train & test rate,

in arcface.py change numbers of ok_test_rate & ng_test_rate.


# Used urls below as a reference

https://github.com/shinmura0/DeepAnomalyDetection_benchmark

https://qiita.com/tom_eng_ltd/items/8d60108b03afe38dff27

https://qiita.com/wakame1367/items/d90fa56bd9d11c4db50e

https://qiita.com/futakuchi0117/items/95c518254185ec5ea485

# New

2020/5/10 Add EfficientNet & ScoreCAM
