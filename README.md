# Studienprojekt-SoSe2025
Entwicklung eines KI-gestützten Prüfvorgangs für die automatisierte Qualitätssicherung additiv gefertigter Bauteile. 

# How to install

- clone repository
```
git clone https://git.noc.ruhr-uni-bochum.de/forchwdd/Studienprojekt-SoSe2025.git
```

- create virtual environment and install requirements

```
cd Studienprojekt-SoSe2025
python -m venv venv
.\venv\Scripts\activate
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

# How to use
To use this repo, you have to add your test data the following way:

    Studienprojekt-SoSe2025/
        └── dataset/
            └── train/
                ├── labels/
                └── images/
            └── test/
                ├── labels/
                └── images/
            └── valid/
                ├── labels/
                └── images/

Augmented Data is added optionally with the augment function as seen in main.py.
This creates an additional folder in data and adds the augmented images into the train file: 

        Studienprojekt-SoSe2025/
            └── dataset/
                └── train/
                    ├── labels/
                    └── images/
                └── train_original/
                    ├── labels/
                    └── images/

Use the main.py to start training.

# Manual Scripts

There are some manual Scripts available. Their usage is not required 
to train a model, but they can help debugging:

- __display_iamge.py:__ display images with their bounding boxes
- __clean_label.py:__ clean labels from label files, so that training can be done
on a single class
- __predict.py:__ predict bounding boxes on selected images

Authors:
Jan Michalak, William Forchap and Shobijan Luxumykanthan
