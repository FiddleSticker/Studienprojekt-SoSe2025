# Studienprojekt-SoSe2025
Entwicklung eines KI-gestützten Prüfvorgangs für die automatisierte Qualitätssicherung additiv gefertigter Bauteile. 

# How to use
To use this repo, you have to add your test data the following way:

    Studienprojekt-SoSe2025/
        └── data/
            └── train/
                ├── labels/
                └── images/
Augmented Data is added optionally with the augment function as seen in main.py.
This creates an additional folder in data and adds the augmented images into the train file: 

        Studienprojekt-SoSe2025/
            └── data/
                └── train/
                    ├── labels/
                    └── images/
                └── aug/
                    ├── labels/
                    └── images/
Use the main.py to start training.

Authors:
Jan Michalak, William Forchap and Shobijan Luxumykanthan
