import os
from jinja2 import Environment, FileSystemLoader


def mkdir(path):
    """
    Creates a directory and makes sure the input path does not correspond to an existing folder.

    Parameters
    ----------
    path : str
        Path where a directory should be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(f'Directory {path} already exists..')


def launch_pipeline(data_aug_name,
                    path_dir_model,
                    path_template_dir,
                    path_predictions,
                    gpu_index=0):
    """
    Launch train / detect / metrics pipeline from odeon.
    Output a folder defined by path_dir_model where will be stored the models (and other files from train)
    and also the metrics and the config files used for the execution of the tools in the pipeline.

    Parameters
    ----------
    data_aug_name : str
        Name of the data augmentation used during the training.
    path_predictions : str
        Path where the predictions created during the detection should be stored.
    path_metrics : str
        Path where the metrics should be stored.
    """
    model_dir = os.path.join(path_dir_model, data_aug_name)
    mkdir(model_dir)

    model_path = os.path.join(model_dir, f"{data_aug_name}.pth")
    if os.path.exists(model_path):
        assert os.path.exists(model_path), f"ERROR: File {model_path} already exist .."

    path_predictions_data_aug = os.path.join(path_predictions, data_aug_name)
    mkdir(path_predictions_data_aug)

    # Load templates and create config files
    template_loader = FileSystemLoader(path_template_dir)
    template_env = Environment(loader=template_loader)
    train_template = template_env.get_template("03_train_template.json")
    detect_template = template_env.get_template("04_detect_template.json")
    metric_template = template_env.get_template("05_metrics_template.json")

    output_train_template = train_template.render(file_name=model_path,
                                                  gpu_index=gpu_index,
                                                  output_path=model_dir)

    output_detect_template = detect_template.render(file_name=model_path,
                                                    output_path=path_predictions_data_aug)

    output_metric_template = metric_template.render(pred_path=path_predictions_data_aug,
                                                    output_path=model_dir)

    templates = [output_train_template, output_detect_template, output_metric_template]

    print(f"Execution of odeon pipeline for {data_aug_name}")
    for idx, tool, template in zip(range(3, 6), ['train', 'detect', 'metrics'], templates):
        json_name = f"{str(idx)}_{tool}_{data_aug_name}.json"
        path_template = os.path.join(model_dir, json_name)
        with open(path_template, "w") as json_file:
            json_file.write(template)
        if tool == 'metrics':  # to launch metrics tool in detach
            os.spawnl(os.P_NOWAIT, f"odeon {tool} -c {path_template}")
        else:
            os.system(f"odeon {tool} -c {path_template}")


if __name__ == '__main__':

    path_dir_model = "/media/hd/speillet/data_augmentation/models/RVB"
    path_template_dir = "/home/dl/speillet/config_jsons/template"
    path_predictions = "/media/hd/speillet/data_augmentation/predictions/RVB"
    data_aug_name = "mstd_rcrop_rot1"

    launch_pipeline(data_aug_name,
                    path_dir_model,
                    path_template_dir,
                    path_predictions)

    # os.system(f"zip -r {data_aug_name}.zip {os.path.join(path_dir_model, data_aug_name)}")
    # os.system(f"mc cp {data_aug_name}.zip s3/speillet/data_aug/models/")
