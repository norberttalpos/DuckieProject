# DuckieProject
1. Datasources:

Links:

For the setup : https://docs.duckietown.org/daffy/AIDO/draft/index.html

GitHub for Training and Data: https://github.com/duckietown/gym-duckietown
                       , and: https://github.com/duckietown/challenge-aido_LF-baseline-behavior-cloning

Data collection is done in the commit method of log_util.py, from where the data of the current step is accessed via a Step data structure. The observation associated with the step is saved as an image, the associated action is inserted in the my_app.txt file, all tagged for clarity.
Once the data is collected, we use detector.py to transform the images into a format more suitable for teaching, filtering out the information of interest (bisector, edge of the road). This is achieved by hsv filtering and by clipping the horizon.
For teaching, we convert the images back to a numpy array format with shape of (window_width, window_height, 3). These images become the input (X) database. The labels (y) are the lines of my_app.txt, these are the actions retrieved from the simulator. From these, we have generated the teaching, validation and test databases for training.



  automatic.py: [<img src="https://colab.research.google.com/assets/colab-badge.svg" width="100"/>](https://colab.research.google.com/drive/17ZmFWd9ipcPhu3UMql5EZ32AyhlOysG2)

  detector.py : [<img src="https://colab.research.google.com/assets/colab-badge.svg" width="100"/>](https://colab.research.google.com/drive/1xQSpIAknsp-DMxFXMpcI60WFaWCoqjEi)
  
  log_util.py : [<img src="https://colab.research.google.com/assets/colab-badge.svg" width="100"/>](https://colab.research.google.com/drive/1kUI_Ohr98yPwcjAsObPhGv1oSjUGq4mM)

  data_process: [<img src="https://colab.research.google.com/assets/colab-badge.svg" width="100"/>](https://colab.research.google.com/drive/1O8lRYQlKN9IQgttoQGnu35wppqE9DZBH)
  
  adatok : [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Google_Drive_icon_%282020%29.svg/1147px-Google_Drive_icon_%282020%29.svg.png" width="20"/>](https://drive.google.com/drive/folders/124WPRwzaz-ePeScy4qqRwlmeeOi_Ii7w?usp=sharing)

