def select_model(model_name):

    if model_name == "mgan":
        import model.mgan.LitModel as model
        return model.MGAN
    else:
        raise f"Unknown model named {model_name}"

def select_dataset(dataset_name, config_file):

    if dataset_name == "ms_coco":
        import dataset.ms_coco_mgan as ms_coco_mgan
        return ms_coco_mgan.MGANCOCODataModule(config_file=config_file)
    else:
        raise f"Unknown model named {dataset_name}"