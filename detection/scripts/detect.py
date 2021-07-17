import os
import mmcv
import pickle
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector


# Specify the path to model config and checkpoint file
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_giou_1x_coco_color/epoch_89.pth'
dataset_folder = 'path/to/your/processed_dataset/folder'  # folder of original image
save_path = 'path/to/your/save/file'  # pickle file with all predictions

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# save the results
dataset_predictions = dict()
for set_name in os.listdir(dataset_folder):
    dataset_predictions[set_name] = dict()
    set_path = os.path.join(dataset_folder, set_name)
    for img in tqdm(os.listdir(set_path)):
        predictions = []
        img_path = os.path.join(set_path, img)
        image = mmcv.imread(img_path)
        result = inference_detector(model, img_path)

        for i, pre in enumerate(result):
            w, h = image.shape[:2]
            for one in pre:
                x1 = one[0]
                y1 = one[1]
                x2 = one[2]
                y2 = one[3]
                conf = one[4]
                predictions.append((w, h, x1, y1, x2, y2, conf, i))
        dataset_predictions[set_name][img] = predictions
    pickle.dump(dataset_predictions, open(save_path, 'wb'))
