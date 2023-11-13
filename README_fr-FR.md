# ODEON

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENCE)


[English](README.md) | Francais

## Pourquoi ce nom
ODEON signifie Object Delineation on Earth Observations with Neural network.

## Quel est le but de cette bibliothèque?
Précédement, la bibliothèque consistait en un ensemble d'outils en ligne de commande pour la segmentation sémantique,
elle est maintenant en train de pivoter vers un cadre agnostique pour l'apprentissage en profondeur appliqué à
l'industrie des SIG.

## Installation
La nouvelle version est toujours en phase de développement avancée, mais vous pouvez toujours
utiliser la version héritée.

### Prérequis d'installation
Comme les dépendances Gdal sont présentes, nous recommandons
d'installer les dépendances via conda/mamba avant d'installer le package :

### Ancienne version
```bash
  git clone -b odeon-legacy git@github.com:IGNF/odeon.git
  cd odeon
  conda (ou mamba) env create -f package_env.yml
  pip install -e .
```

#### Nouvelle version
```bash
  git clone git:odeon-legacy git@github.com:IGNF/odeon.git
  cd odeon/packaging
  conda (ou mamba) env create -f package_env.yaml
  pip install -e .
```
