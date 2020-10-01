# Detect : 

**La commande detect.py effectue une prédiction (inférence) d'un réseaux de neurones sur patchs ou sur une zone d'intérêt**


La commande de base pour faire de la détection est :

* `python detect.py /chemin/du/<fichier_configuration>.json`  

un exemple de fichier de configuration est 

```json
{
  "sources": {
    "RVB": "https://wxs.ign.fr/$CLE_GEOPORTAIL$/geoportail/r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES=normal&LAYERS=ORTHOIMAGERY.ORTHOPHOTOS.ORTHO-EXPRESS.2016",
    "IRC": "https://wxs.ign.fr/$CLE_GEOPORTAIL$/geoportail/r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES=normal&LAYERS=ORTHOIMAGERY.ORTHOPHOTOS.IRC.2016",
    "MNS": "/media/HP1710W030/DATA_CHALLENGE/etude_OCS/bati/85/vrt/mns2016.vrt",
    "MNT": "https://wxs.ign.fr/$CLE_GEOPORTAIL$/geoportail/r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES=normal&LAYERS=RGEALTI-MNT_PYR-ZIP_FXX_LAMB93_WMS"
  },
  "zone_shp": "/mnt/data/etude-ocs/nice/emprise_nicolas/EMPRISE_CHANTIER_QUADRI.shp",
  "image": {
    "side": 256,
    "pixel_size": 0.2,
    "marge_pixel": 30
  },
  "model": {
    "type": "deeplab",
    "filename": "/mnt/data/etude-ocs/mono_deeplab_mobilenetv2/deeplab_pytorch_137_valloss_0.075.pth"
  },
  "output_path": "/mnt/data/etude-ocs/nice",
  "tile_factor": 4,
  "num_workers": 3,
  "use_gpu": 0,
  "batch_size": 2,
  "out_dalle_size_km": 5,
  "out_gdal_type": "Byte",
  "epsg": 2154,
  "export_input": 1,
  "channels": 5,
  "withmns": 1,
  "no_of_classes": 1
}
```

Celui-ci est composé de plusieurs parties:

 * la définition des données raster qui servent de données sources aux réseau de neurones
 * la définition du type de detection et de la zone d'intérêt
 * la définition du modèle de réseau de neurones à utiliser pour la prédiction
 * la configuration du type des données en sorties
 * la configuration des paramètres de calculs (performances)
 
## Source de données

La définition des sources de donnés est faites dans le paramètres sources.

```json
{
  "sources": {
    "RVB": "https://wxs.ign.fr/$CLE_GEOPORTAIL$/geoportail/r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES=normal&LAYERS=ORTHOIMAGERY.ORTHOPHOTOS.ORTHO-EXPRESS.2016",
    "IRC": "https://wxs.ign.fr/$CLE_GEOPORTAIL$/geoportail/r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES=normal&LAYERS=ORTHOIMAGERY.ORTHOPHOTOS.IRC.2016",
    "MNS": "/media/HP1710W030/DATA_CHALLENGE/etude_OCS/bati/85/vrt/mns2016.vrt",
    "MNT": "https://wxs.ign.fr/$CLE_GEOPORTAIL$/geoportail/r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES=normal&LAYERS=RGEALTI-MNT_PYR-ZIP_FXX_LAMB93_WMS"
  },
  "image": {
    "pixel_size": 0.2
  },
```

Pour l'instant les quatres sources de données: RVB, IRC, MNS, MNT doivent être spécifiées même si le model utilisé
n'utilise que 3 ou 4 des 5 bandes d'information disponibles

Les sources de données peuvent pour l'instant être de type vrt ou bien flux WMS.


## Modèles

La configuration du modèle à utiliser est faite avec les paramètres suivant

``` json
  "model": {
    "type": "unet",
    "filename": "/home/dlsupport/tmp/models/bati_34_valloss1000_522.pth"
  },
  "use_gpu": 0
```

Pour l'instant mode/type peut prendre les valeurs suivantes:
 
* unet (<400k paramètres) : transcription du modèle préalablement utilisé sous tensorflow/keras
* heavyunet (>11M paramètres) : modèle conforme à la publication originelle avec une amélioration (*batch normalisation*)
* deeplab

Pour plus d'information, se référer au wiki du projet.


**model/filename**:
chemin du fichier *.pth contenant les poids du model. Un fichier json du même nom doit normalement accompagner ce fichier

**use_gpu**:
option 0/1 afin d'indiquer si la prédiction doit utiliser le GPU ou seulement le CPU. Dans le cas ou le téléchargement
des données est ce qui prend le plus de temps et ou le model est simple, l'utilisation du GPU n'apporte pas forcément
grand chose. 

## Détection sur échantillons

La phase de détection sur path/échantillons a besoin :

* des images sur lesquelles générer des masques de détection (**input_path** et les sous répertoires de département **input_path_subdirs**).
On précise ensuite dans ces départements quels types d'échantillons on souhaite (**class_subdirs**).

Les masques seront sauvegardés sous forme de raster 32bits dans **output_path** avec la même arborescence.

> Attention : le contenu entier des répertoire **class_subdirs** va être utilisé.

``` json
  "description": "template pour la detection sur des images (pas de flux)",
  "output_path": "/home/dlsupport/tmp/detection/",
  "input_path": "/media/HP1710W028/DATA_CHALLENGE/etude_OCS/images_communes/",
  "input_path_subdirs": [
    "31/2016/test/"
  ],
  "class_subdirs": [
    "bati/",
    "autres/"
  ],
```

La seconde partie donne le chemin du modèle à utiliser. Il est important que le fichier **json** utilisé lors de l'entrainement
porte le même nom que le modèle (extention *pth*).


## Détection sur zones géographiques / région d'intérêt

### zone d'interêt

la zone d'intérêt est définie dans le paramètre **zone_shp** qui pointe vers un fichier Shapefiel contenant une géométrie
de type polygone définissant l'emprise géographique sur laquelle on souhaite lancer la prédiction du model


```json
  "zone_shp": "/mnt/data/etude-ocs/nice/emprise_nicolas/EMPRISE_CHANTIER_QUADRI.shp"
```

### format de sortie

Le format de prédiction pour la détection zone peut être de deux type :
  
  * par ensemble de tuiles réprésentant un nombre entier de patch de detection (image/side)
  * par ensemble de dalles kilométriques.

Pour les tuiles de patch l'option principale est tile_factor qui doit être un entier. La taille d'une tuile de prédiction est
alors de tile_factor*image/side.  Le réseau fde neurones traite donc tile_factor^2 patchs de detection en même temps (un
peu plus en cas de recouvrement entre detection)

```json
  "output_path": "/mnt/data/etude-ocs/nice",
  "tile_factor": 4
```
Les tuiles en sorties ont un prefix "predict" et sont nommées avec les coordonées du coin haut gache de l'image exprimées
en mètre.

Pour les tuiles kilometriques, leur taille en sortie est définie par

```json
  "out_dalle_size_km": 5
```
la taille est exprimée en kilomètre. Les tuiles en sorties ont un prefix "predict" et sont nommées avec les coordonées
du coin haut gache de l'image exprimées en kilomètre et avec une taille de quatre (coordonnée en X peut commencer par
zéro). De plus les dalles commencent sur des kilomètres entiers.


**marge pixel**

De plus la prédiction par zone utilise le paramètre image/marge_pixel qui définie le nombre de pixel en bord de zone qui
ne serviront pas à la construciton de la prédiction finale car la detection en bord de patch est moins bonne que celle en
centre de patch.

**format pixel**

La prédiction peut être soit de type "Float32" avec des valeurs entre [0,1] soit de type "Byte" avec des valeurs entre
0 et 255.

**export image rvb+ir+mne**

en plus de la prédiction il est possible d'exporter les images 5 canaux en float32 servant à faire la prédiction en
mettant le paramètre export_input à 1.

Cela peut être utile pour analyser une prédiction en vérifiant que les données
fournies sont correctes et également pour vérifier la source d'éventuel problème de géoréférencement.
*WARNING*: si l'export des images est activé avec une grande taille de tuile kilométrique cela peut conduire à
créer des images Tiff de plus de 4Go et peut faire planter gdal/la detection.

## Paramètres du calcul

**tuilage / dallage de la région d'intérêt**

la zone d'intérêt est dallée par un ensemble de tuiles qui sera fournis par ordre d'index au code d'inférence du réseau
de neurone. Ce tuillage/dallage comprend trois paramètres essentiels

 * out_dalle_size_km: est-ce que l'on doit fournir des tuiles selon un dallage kilométriques ou non
 * tile_factor: la taille d'une extraction image élémentaire (en multiple de image size). Permet de ne pas faire trop
  de requêtes de petites zones ni de faire des requêtes de trop grosses zones (2km par 2km) par exemple
 * marge_pixel: définis le recouvrement à utiliser entre tuile de requêtes images et aussi entre dalles kilmétriques.

Dans le cas d'une prédiction par zone kilométriques le dallage est fait en deux temps et selon une hierarchie
à deux niveaux:

  * Dans un premier temps la zone d'intêret est tuilée selon un carroyage kilométrique de la taille indiquée par
    out_dalle_size_km et avec un recouvrement définie par marge_pixel*pixel_size. Le reouvrement est pris en dehors des 
    tuiles kilometriques et déborde donc des tuiles kilométriques initiales.
  * Dans un deuxième temps chaque dalle kilométriques est tuilées par des tuiles des tile_factor*(image side) et toujours
    en prenant en compte le recouvrement demandé pour les prédictions. Ici les sous-tuiles sont comprises dans les tuiles
    kilométriques donc celles des dernières lignes/colonnes peuvent avoir un recouvrement plus important que celui demandé.
 
**!!le tuilage de la zone est exporté en fichiers shapefile!!**
Afin de comprendre quelles sont les tuiles images extraites et dans quels ordres elles sont fournies au réseau quatres
fichiers shapefile d'emprise de tuile sont créés:

* save_tile_no_overlap.shp : limite des tuiles de prédicition sauvegardées sans recouvrement 
* save_tile_overlap.shp: limites des tuiles images kilométriques extraites et correspondant à une prédiction avec recouvrement
* load_tile_no_overlap.shp: limite des tuiles extraites des sources raster et fournies à la prédiction du réseau sans recouvrement
* load_tile_overlap.shp: limite des tuiles extraites des sources raster et fournies à la prédiction du réseau avec recouvrement

**extraction des donnés/parallélisation **

L'extraction des données est faites sous forme d'un DataLoader pytorch et en possède les paramètres standards

 * num_workers: nombre de processus en parallèle chargé de l'extraction des données image à partir des données sources (WMS, vrt)
 * batch_size: taille du batch de tuile/nombre de tuiles extraites fournies en même temps à la prédiction du réseau.
 
 En plus dans la detectio zone :
 
  * tile_factor: paramètre de coefficient multiplicateur permettant de définir la tailles des images qui seront requêtes
    aux sources images (WMS ou VRT)

Le début de la detection peut donc être lent car elle ne commencent que quand num_workers*batch_size*tile_factor^2 imagettes
de tailles sample_size ont été récupérées.

**detection par réseau**

La detection est lancé indépendament sur chaque image de tailles tile_factor*sample_size. Par defaut la detection est
faites avec des patchs de taille sample_size (l'image en entrée est redécoupé). Toutefois cette taille peut être
augmenté en spécifiant le paramètre detect_size.
  
  * detect_size : taille des patchs founis au réseau pytorch pur la detection; Taille en pixel 

## TODO 


les paramètres suivant ne sont pas encore utilisés (anciens paramètres à proter sur les nouveaux codes)
 
**channels**
  
**withmns**

**no_of_classes**


## workflow /methodo detection by zone
