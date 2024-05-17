def select_model(model_name):

    if model_name == "bsc":
        import model.bsc.LitModel as model
        return model.BSC
    else:
        raise f"Unknown model named {model_name}"

def select_dataset(dataset_name, config_file):

    if dataset_name == "ms_coco":
        import dataset.ms_coco_bsc as ms_coco_bsc
        return ms_coco_bsc.BSCCOCODataModule(config_file=config_file)
    else:
        raise f"Unknown model named {dataset_name}"