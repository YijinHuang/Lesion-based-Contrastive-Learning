import os
import pickle
from tqdm import tqdm


predictions = pickle.load(open('EyePACS_predictions_color_with_image_size.pkl', 'rb'))  # pickle file from detect.py
dataset_folder = 'path/to/your/processed_dataset/folder'  # folder of original image
save_path = 'path/to/your/save/file'  # pickle file for training
confidence_threshold = 0.5


dataset_predictions = dict()
for set_name in os.listdir(dataset_folder):
    dataset_predictions[set_name] = dict()
    set_path = os.path.join(dataset_folder, set_name)
    for img in tqdm(os.listdir(set_path)):
        # img_path = os.path.join(set_path, img)
        lesions = []
        for lesion in predictions[set_name][img]:
            w, h, x1, y1, x2, y2, conf, le = lesion
            w_ratio = 512 / w
            h_ratio = 512 / h
            x1, x2 = x1 * w_ratio, x2 * w_ratio
            y1, y2 = y1 * h_ratio, y2 * h_ratio
            b_w, b_h = x2 - x1, y2 - y1
            if b_w <= 128 and b_h <= 128 and conf > confidence_threshold and (x1 > 5 and x2 < 512 - 5 and y1 > 5 and y2 < 512 - 5):
                lesions.append((x1, y1, x2, y2))
        if lesions:
            dataset_predictions[set_name][img] = lesions

pickle.dump(dataset_predictions, open('./EyePACS_predictions_128_05.pkl', 'wb'))
