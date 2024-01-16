from PIL import Image
import pandas as pd
import os
import xmltodict
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")


def load_image(file_name,y):
  raw = tf.io.read_file(file_name)
  tensor = tf.image.decode_image(raw)
  tensor = tf.image.crop_to_bounding_box(tensor,offset_width=0,offset_height=25,target_width=178,target_height=178)
  tensor = tf.cast(tensor, tf.float32) / 255.0
  return tensor,y



def load_dataset_voc(path, path_to_images):
    
    classes = ["person", "bird", "cat", "cow", "dog", "horse", "sheep",
               "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", 
               "bottle", "chair", "dining table", "potted plant", "sofa", "tv/monitor"]
    
    images = []
    targets = []
    for file in os.listdir(path):
        with open(path+file,"r") as f:
            file_content = f.read()
        file_content = xmltodict.parse(file_content)

        image_file_name = file_content["annotation"]["filename"]
        objects_in_image = file_content["annotation"]["object"]
        if not type(objects_in_image) == list:
            objects_in_image =  [objects_in_image]

        objects_in_image = [obj["name"] for obj in objects_in_image]

        target = [1 if c in objects_in_image else 0 for c in classes]
        

        image = Image.open(path_to_images + image_file_name)
        image=image.resize((224,224))
        images.append(tf.keras.utils.img_to_array(image))
        targets.append(target)

    return images,targets


def load_dataset_celeb_a(path, path_to_images):
    csv = pd.read_csv(path)

    list_image_paths = []
    list_image_attributes = []
    for elem in csv.iloc:
        values = elem.values
        image_name = values[0]
        image_attributes = values[1:]
        image_attributes = [1 if attr == 1 else 0 for attr in image_attributes]

        list_image_paths.append(path_to_images+image_name)
        list_image_attributes.append(image_attributes)

    return list_image_paths,list_image_attributes

