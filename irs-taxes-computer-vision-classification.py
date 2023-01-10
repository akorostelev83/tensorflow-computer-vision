import locale
import os

import ghostscript
##### must install ghostscript exe on windows https://ghostscript.com/releases/gsdnld.html #####
import numpy
from keras.utils import np_utils
from PIL import Image
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report


kaggle_w2_folder = 'C:/Users/SUPREME/Documents/SurePrep/w2_samples_multi_clean/'
kaggle_w2_images_folder = 'C:/Users/SUPREME/Documents/SurePrep/w2_samples_multi_clean_images/'

def download_kaggle_dataset():
    os.environ['KAGGLE_USERNAME'] = 'andreik1000'
    os.environ['KAGGLE_KEY'] = 'ef0ff9173d70f29a62f46a665a5b4d1b'

    from kaggle.api.kaggle_api_extended import KaggleApi

    dataset = 'mcvishnu1/fake-w2-us-tax-form-dataset'
    path = 'datasets/fake_w2'

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_file(dataset, 'W2_Multi_Clean_DataSet_02.ZIP', path)

def extract_kaggle_dataset():
    from zipfile import ZipFile
    file_name = "C:/datasets/fake_w2/W2_Multi_Clean_DataSet_02.ZIP"
    os.chdir('C:/Users/SUPREME/Documents/SurePrep/')

    with ZipFile(file_name, 'r') as zip:	
        zip.printdir()	
        print('Extracting all the files now...')
        zip.extractall()
        print('Done!')

def convert_pdfs_to_images():
    def pdf2jpeg(pdf_input_path, jpeg_output_path):
        args = ["pef2jpeg", # actual value doesn't matter
                "-dNOPAUSE",
                "-sDEVICE=jpeg",
                "-r144",
                "-sOutputFile=" + jpeg_output_path,
                pdf_input_path]

        encoding = locale.getpreferredencoding()
        args = [a.encode(encoding) for a in args]

        ghostscript.Ghostscript(*args)

    def get_pdf_files():
        return os.listdir(kaggle_w2_folder) 

    [pdf2jpeg(
        kaggle_w2_folder+pdf_path,
        kaggle_w2_images_folder+pdf_path.replace('pdf','jpeg')) 
            for pdf_path in get_pdf_files() if pdf_path.endswith(".pdf")]

def get_class_arrays():
    print('sorting w2 class files into arrays')
    adp1s,adp2s,irs1s,irs2s = [],[],[],[]
    for path in os.listdir(kaggle_w2_images_folder):    
        if 'adp1' in path.lower():
            adp1s.append(kaggle_w2_images_folder+path)
        if 'adp2' in path.lower():
            adp2s.append(kaggle_w2_images_folder+path)
        if 'irs1' in path.lower():
            irs1s.append(kaggle_w2_images_folder+path)
        if 'irs2' in path.lower():
            irs2s.append(kaggle_w2_images_folder+path)
    assert len(adp1s) == 250
    assert len(adp2s) == 250
    assert len(irs1s) == 250
    assert len(irs2s) == 250    
    return adp1s,adp2s,irs1s,irs2s

def get_training_features_labels(training_files):
    features = []

    for training_file in training_files:
        features.append(
            numpy.array(
                Image.open(training_file))\
                    .astype(numpy.float32)/255.0)

    X_train = numpy.array(features).reshape(len(features),1584,1224,3)
    y_train = np_utils.to_categorical(
        numpy.array([0 for i in range(0,25)] + [1 for i in range(0,25)] + [2 for i in range(0,25)] + [3 for i in range(0,25)]))

    assert len(X_train) == len(y_train)
    assert len(X_train) == 100

    return X_train, y_train

def get_test_features_labels(test_files):
    features = []

    for test_file in test_files:
        features.append(
            numpy.array(Image.open(test_file))\
                .astype(numpy.float32)/255.0)

    X_train = numpy.array(features).reshape(len(features),1584,1224,3)
    y_train = np_utils.to_categorical(
        numpy.array([0 for i in range(0,25)] + [1 for i in range(0,25)] + [2 for i in range(0,25)] + [3 for i in range(0,25)]))

    assert len(X_train) == len(y_train)
    assert len(X_train) == 100

    return X_train, y_train

def get_label(image_path):
    image_path = image_path.lower()
    
    if 'adp1' in image_path:
        return 0
    if 'adp2' in image_path:
        return 1
    if 'irs1' in image_path:
        return 2
    if 'irs2' in image_path:
        return 3

    raise Exception(f'label not found in {image_path}')

def get_predictions_and_truth(model,_training_files):
    print('making predictions')
    y_predictions = []
    y_true = []

    for path in os.listdir(kaggle_w2_images_folder):
        if kaggle_w2_images_folder + path in _training_files:
            print(f'skipping image used for training: {path}')
            continue
        w2_features = numpy.array(
            Image.open(kaggle_w2_images_folder + path))\
                .reshape(1,1584,1224,3)\
                    .astype(numpy.float32)/255.0
        #print(f'running prediction on image: {path}')
        y_predictions.append(model.predict(w2_features)[0])
        y_true.append(get_label(path))

    assert len(y_predictions) == len(y_true)
    y_true_categorical = np_utils.to_categorical(y_true)
    return y_true_categorical,numpy.array(y_predictions)

def get_model(num_classes):
    print(f'number of classes in tf.kears: {num_classes}')
    print('preping model')
    model = tf.keras.Sequential([       
        tf.keras.layers.Dropout(0.25,input_shape=(1584,1224,3)),
        tf.keras.layers.Conv2D(16,5,5, padding='same', activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.MaxPooling2D((2,2), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')])

    print('compiling model')
    model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy'])

    return model

def get_training_test_file_paths(): 
    adp1s,adp2s,irs1s,irs2s = get_class_arrays()
    training_files = adp1s[:25] + adp2s[:25] + irs1s[:25] + irs2s[:25]
    test_files = adp1s[26:51] + adp2s[26:51] + irs1s[26:51] + irs2s[26:51]
    return training_files,test_files

def print_model_results(y_true_categorical,y_predictions):
    y_true_categorical = y_true_categorical.astype('int32')
    y_predictions = numpy.round(numpy.array(y_predictions))
    
    print('getting model stats')
    metric = tf.keras.metrics.CategoricalAccuracy()
    metric.update_state(y_true_categorical,y_predictions)
    print('categorical accuracy')
    print(metric.result().numpy())

    accuracy_score_ = accuracy_score(y_true_categorical, y_predictions)
    print(f"Test Accuracy : {accuracy_score_}")

    print("Classification Report :")
    print(classification_report(
        y_true_categorical, 
        y_predictions, 
        target_names=['adp1','adp2','irs1','irs2']))

#print('downloading kaggle dataset as zip')
#download_kaggle_dataset()
#print('extracting data from zipped file')
#extract_kaggle_dataset()
#print('converting pdfs to images')
#convert_pdfs_to_images()

training_files,test_files = get_training_test_file_paths()
X_train, y_train = get_training_features_labels(training_files)
X_test, y_test = get_test_features_labels(test_files)

num_classes=y_train.shape[1]

model = get_model(num_classes)

print('training model')
model.fit(
    X_train, 
    y_train, 
    batch_size=16,
    epochs=25,
    verbose=1,
    validation_data=(X_test, y_test))

y_true_categorical,y_predictions= get_predictions_and_truth(model,training_files)

print_model_results(
    y_true_categorical,
    y_predictions)