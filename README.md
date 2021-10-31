# DuckieProject
1. Datasources:

Links:

For the setup : https://docs.duckietown.org/daffy/AIDO/draft/index.html

GitHub for Training and Data: https://github.com/duckietown/gym-duckietown
                       , and: https://github.com/duckietown/challenge-aido_LF-baseline-behavior-cloning

Az adatok begyűjtését a log_util.py commit metódusában végezzük, ahonnan elérjük az aktuális lépés adatait egy Step adatstruktúrán keresztül. A lépéshez tartozó observation-t képként elmentjük, a hozzá tartozó action-t a my_app.txt file-ba szúrjuk be, mindezt címkével ellátva az egyértelműség érdekében.
Az adatgyűjtést követően a detector.py segítségével transzformáljuk a képeket egy tanításra alkalmasabb formátumra, kiszűrve a számunkra érdekes információt (felezővonal, út széle). Ezt hsv filtering-el, illetve a horizont levágásával érjük el.
Tanításhoz visszakonvertáljuk a képeket numpy array formátumba, aminek shape-je (window_width, window_height, 3). Ezek a képek lesznek az input (X) adatbázis. A címkék (y) a my_app.txt sorai, vagyis a szimulátorból szerzett action-ök. Ezekből előállítottuk a tanításhoz szükséges tanító, validációs és teszt adatbázisokat.


  automatic.py: [<img src="https://colab.research.google.com/assets/colab-badge.svg" width="100"/>](https://colab.research.google.com/drive/17ZmFWd9ipcPhu3UMql5EZ32AyhlOysG2)

  detector.py : [<img src="https://colab.research.google.com/assets/colab-badge.svg" width="100"/>](https://colab.research.google.com/drive/1xQSpIAknsp-DMxFXMpcI60WFaWCoqjEi)
  
  adatok : [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Google_Drive_icon_%282020%29.svg/1147px-Google_Drive_icon_%282020%29.svg.png" width="20"/>](https://drive.google.com/drive/folders/124WPRwzaz-ePeScy4qqRwlmeeOi_Ii7w?usp=sharing)

