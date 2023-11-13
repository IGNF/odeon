IS_TEST_COL = "test"
IMAGENET = {"mean": [0.485, 0.456, 0.406], "std": [0.228, 0.224, 0.225]}
N_URBAIN_CLASS = 9  # 8 + 1 background (0 value)
N_NAF_CLASS = 16  # 15 + 1 background (0 value)
UINT8_MAX = 255.0
test_nom = {"nomenclature_name": [{"name": "batiment", "class": 1, "rgb": [219, 14, 154], "key": 1}]}
NOMENCLATURE = {"urbain": {0: [[0, 0, 0], "nolabel", 0],
                           1: [[219, 14, 154], "batiment", 1],
                           2: [[16, 140, 57], "ligneux", 3],
                           3: [[200, 253, 146], "herbacee", 3],
                           4: [[248, 12, 0], "bitume", 1],
                           5: [[190, 178, 151], "mineraux", 2],
                           6: [[220, 174, 74], "sol_nus", 2],
                           7: [[21, 83, 174], "eau", 2],
                           8: [[61, 230, 235], "piscine", 1]},
                "naf": {0: [[0, 0, 0], "nolabel", 0],
                        1: [[219, 14, 154], "batiment", 1],
                        2: [[147, 142, 123], "zone_permeable", 1],
                        3: [[248, 12, 0], "zone_impermeable", 1],
                        4: [[31, 230, 235], "piscine", 1],
                        5: [[219, 14, 154], "sol_nus", 2],
                        6: [[169, 112, 1], "surface_eau", 2],
                        7: [[21, 83, 174], "eau", 2],
                        8: [[25, 74, 38], "coniferes", 3],
                        9: [[138, 179, 160], "coupe", 3],
                        10: [[70, 228, 131], "feuillus", 3],
                        11: [[243, 166, 13], "broussaille", 3],
                        12: [[21, 83, 174], "vigne", 3],
                        13: [[255, 243, 13], "culture", 3],
                        14: [[228, 223, 124], "terre_labouree", 3],
                        15: [[193, 62, 236], "other", 0]
                        }
                }

URBAIN_COLOR_MAP = [value[0] for _, value in NOMENCLATURE["urbain"].items()]
NAF_COLOR_MAP = [value[0] for _, value in NOMENCLATURE["naf"].items()]
