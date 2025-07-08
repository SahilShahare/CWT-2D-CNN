# CWT-2D CNN
This project involves detecting Atrial Fibrillation using Continuous Wavelet Transform(CWT) and 2D-Convolutional Neural Network(CNN) based on work.

**Original Paper**: [He R, Wang K, Zhao N, Liu Y, Yuan Y, Li Q, Zhang H. Automatic Detection of Atrial Fibrillation Based on Continuous Wavelet Transform and 2D Convolutional Neural Networks. Front Physiol. 2018 Aug 30;9:1206. doi: 10.3389/fphys.2018.01206. PMID: 30214416; PMCID: PMC6125647.](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2018.01206/full)

## 🔧 Configuration

Replace dummy paths in `paths.json` file in the project directory with paths to the datasets on your system:

```json
{
    "mit-bih-arrhythmia": "/absolute/path/to/mit-bih-arrhythmia",
    "incart": "/absolute/path/to/incartdb",
    "mit-bih-supra-ventricular": "/absolute/path/to/svdb"
}
```

## ⚙️ Requirements & Installation
```
Python 3.12.8
wfdb 4.3.0
scipy 1.16.0
numpy 2.2.1
Pywavelets 1.8.0
scikit-learn 1.7.0
pandas 2.3.0
opencv-python 4.11.0.86
torch 2.5.0
torchvision 0.20.0
torchsummary 1.5.1
```
Ensure ```python >= 3.12.8``` is installed. 

```
conda activate <your env>
pip install -r requirements.txt
```

## 🚀 Execution
### Random-Split
```
cd RandomSplit
python3 data_extract.py
python3 train.py
```
Find the results in ```RandomSplit/logs/{yyyy-mm-dd_hh:mm:ss}.txt``` file.

### Patient Segregated Split
```
cd Patient Segregated Split
python3 data_extract.py
python3 train.py
```
Find the results in ```Patient Segregated Split/logs/{yyyy-mm-dd_hh:mm:ss}.txt``` file.

### Cross Dataset Split

**Train: AFDB & LTAFDB / Test: SHDB**
```
cd InterDataset/
python3 data_extract.py
python3 train_afdb_ltafdb_shdb.py
```
Find the results in ```Inter Dataset/logs/afdb_ltafdb_shdb/{yyyy-mm-dd_hh:mm:ss}.txt``` file.

**Train: AFDB & SHDB / Test: LTAFDB**
```
cd InterDataset/
python3 data_extract.py
python3 train_afdb_shdb_ltafdb.py
```
Find the results in ```Inter Dataset/logs/afdb_shdb_ltafdb/{yyyy-mm-dd_hh:mm:ss}.txt``` file.

**Train: SHDB & LTAFDB / Test: AFDB**
```
cd InterDataset/
python3 data_extract.py
python3 train_shdb_ltafdb_afdb.py
```
Find the results in ```Inter Dataset/logs/shdb_ltafdb_afdb/{yyyy-mm-dd_hh:mm:ss}.txt``` file.

## 💡 Tips
* Use Screen to run program. [more info.](https://www.geeksforgeeks.org/linux-unix/screen-command-in-linux-with-examples/)
  ```
  screen -S <your-session-name>
  conda activate <your-env>
  pip install -r requirements.txt
  .
  .
  <run-your code>
  .
  .
  
  ```
