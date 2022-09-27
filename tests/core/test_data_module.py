# from cd_ortho.core.data_module import Input

change_dataset: str = "/media/DATA10T_3/gers/change_dataset/dataset_v1.shp"
segmentation_dataset: str = "/media/HP-2007S005-data/gers/" \
                            "supervised_dataset/supervised_dataset_with_stats_and_weights.geojson"

input_fields = {"image": {"name": "image", "type": "image", "dtype": "uint8"},
                "mask": {"name": "mask", "type": "image", "encoding": "integer"}
                }


def test_data_module_creation():
    # data_module = Input(input_fields=input_fields,
    # input_fit_file=segmentation_dataset, input_validate_file=segmentation_dataset)
    ...
