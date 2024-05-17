def select_model(model_name):

    if model_name == "mit":
        import model.mit.LitModel as model
        return model.MIT
    else:
        raise f"Unknown model named {model_name}"

def select_dataset(dataset_name, config_file):

    if dataset_name == "ms_coco":
        import dataset.ms_coco_mit as ms_coco_mit
        # import dataset.visual_genome as ms_coco_mit
        return ms_coco_mit.MITCOCODataModule(config_file=config_file)
    else:
        raise f"Unknown model named {dataset_name}"