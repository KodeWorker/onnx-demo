call "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\bin\setupvars.bat"

set MODEL=./openvino_models/efficientnet-b0/opt-efficientnet-b0.xml
set INPUT=D:\Datasets\Kaggle\dogs-vs-cats\train\cat\cat.1.jpg
set DEVICE=MYRIAD
set LABELS=./imagenet.labels
set TOPN=5

python openvino_classification_sample.py ^
-m %MODEL% ^
-i %INPUT% ^
-d %DEVICE% ^
--labels %LABELS% ^
-nt %TOPN%
