from django.shortcuts import render
from input_info.forms import InfoForm
from input_info.getSimilarPeople import getPeople
from shutil import copyfile
import shutil
import os
import cv2

import math
from sklearn import neighbors
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from django.http import HttpResponseRedirect
import pandas as pd
import urllib.request

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
number = "0"
data = [{},{}]
dataset = pd.DataFrame(data)


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.
    :param train_dir: directory that contains a sub-directory for each known person, with its name.
     (View in source code to see train_dir example tree structure)
     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(img_path, predictions):
    """
    Shows the face recognition results visually.
    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()





def ask_info(request):
    SITE_ROOT = os.path.dirname(os.path.realpath(__file__))
    
    form = InfoForm(request.POST or None)
    #Als het form is ingevuld en wordt gesubmit worden de ingevoerde gegevens opgehaald en opgeslagen in variabelen
    if request.method == "POST":
        if form .is_valid():
            age = form .cleaned_data["age"]
            gender = form .cleaned_data["gender"]
            race = form .cleaned_data["race"]
            sexorien = form .cleaned_data["sexorien"]
            genid = form .cleaned_data["genid"]
            #Haal een dataset op van alle mensen die dezelfde eigenschappen hebben en sla ze op in een globale variabele
            df4 = getPeople(age, gender, race, sexorien, genid)
            global dataset
            dataset = df4
            #Verwijder alle foto's in de "train" folder die er mogelijk nog van de vorige keer stonden
            folder = SITE_ROOT + r"\knn_examples\train"
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
            #Kopieer de foto's van alle mensen uit de dataset naar de "train"folder"        
            for index, row in df4.iterrows():
                src = SITE_ROOT + r"\knn_examples\all_images\1 (" + str(row['Nr']) + r")\1 (" + str(row['Nr']) + r").jpg"
                newpath = SITE_ROOT + r"\knn_examples\train\1 (" + str(row['Nr']) + r")"
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                dst = SITE_ROOT + r"\knn_examples\train\1 (" + str(row['Nr']) + r")\1 (" + str(row['Nr']) + r").jpg"
                urllib.request.urlretrieve(r"https://raw.githubusercontent.com/Darth-Fedor/Webapp/main/webapp/input_info/knn_examples/all_images/1%20(" + str(row['Nr']) + r")/1%20(" + str(row['Nr']) + r").jpg", dst)
                #copyfile(src, dst)
                
            #Maak een foto via de webcam     
            key = cv2. waitKey(1)
            webcam = cv2.VideoCapture(1)
            while True:
                try:
                    check, frame = webcam.read()
                    #print(check) #prints true as long as the webcam is running
                    #print(frame) #prints matrix values of each framecd 
                    cv2.imshow("Capturing", frame)
                    key = cv2.waitKey(1)
                    if key == ord('s'): 
                        cv2.imwrite(filename=SITE_ROOT + '/knn_examples/test/saved_img.jpg', img=frame)
                        webcam.release()
                        img_new = cv2.imread(SITE_ROOT + '/knn_examples/test/saved_img.jpg', cv2.IMREAD_GRAYSCALE)
                        img_new = cv2.imshow("Captured Image", img_new)
                        cv2.waitKey(1650)
                        cv2.destroyAllWindows()
                        
                        print("Image saved!")
                        # STEP 1: Train the KNN classifier and save it to disk
                        # Once the model is trained and saved, you can skip this step next time.
                        print("Training KNN classifier...")
                        classifier = train(SITE_ROOT + "/knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
                        print("Training complete!")
                    
                        # STEP 2: Using the trained classifier, make predictions for unknown images
                        for image_file in os.listdir(SITE_ROOT + "/knn_examples/test"):
                            full_file_path = os.path.join(SITE_ROOT + "/knn_examples/test", image_file)
                    
                            print("Looking for faces in {}".format(image_file))
                    
                            # Find all people in the image using a trained classifier model
                            # Note: You can pass in either a classifier file name or a classifier model instance
                            predictions = predict(full_file_path, model_path="trained_knn_model.clf")
                    
                            # Print results on the console
                            for name, (top, right, bottom, left) in predictions:
                                print("- Found {} at ({}, {})".format(name, left, top))
                                #Als de foto niet herkend wordt en er unkown uitkomt, wordt er doorverwezen naar de "no_result" pagina
                                if name=="unknown":
                                    return HttpResponseRedirect('/survey/no_result')
                                #Als de foto wel wordt herkend wordt het nummer van de persoon opgeslagen en de foto wordt gekopiëerd naar de "result" folder
                                global number
                                number = name[3:-1]
                                src = SITE_ROOT + r"\knn_examples\all_images\1 (" + number + r")\1 (" + number + r").jpg"
                                dst = SITE_ROOT + r"\knn_examples\result\result.jpg"
                                copyfile(src, dst)
                                
                            #Ga naar de pagina met het resultaat
                            return HttpResponseRedirect('/survey/result')
                        
                        
                        break
                    elif key == ord('q'):
                        print("Turning off camera.")
                        webcam.release()
                        print("Camera off.")
                        print("Program ended.")
                        cv2.destroyAllWindows()
                        break
                    
                except(KeyboardInterrupt):
                    print("Turning off camera.")
                    webcam.release()
                    print("Camera off.")
                    print("Program ended.")
                    cv2.destroyAllWindows()
                    break
                                  
    context = {"form": form }
    return render(request, "input_info.html", context)


def show_result(request):
    global number    
    global dataset
    #Laat het resultaat zien van de persoon met het juiste nummer
    for index, row in dataset.iterrows():
        if(str(row['Nr']) == number):
            name = row["Name"]
            age = row["Age"]
            gender = row["Gender"]
            rank = row["Rank"]
            department = row["Department"]
            function = row["Function"]
            race = row["Race"]
            sexualOrientation = row["Sexual_orientation"]
            genderIdentity = row["Gender_identity"]
            story = row['Story']
    
    base_image = "http://127.0.0.1:8000/input_info/knn_examples/result/result.jpg"
    
    context = {'base_image': base_image, "name": name, "function": function, "story": story, "age": age, "gender": gender ,"rank": rank, "department": department, "race": race, "sexualOrientation": sexualOrientation, "genderIdentity": genderIdentity}
    return render(request, "result.html", context)



def home(request):
   
    return render(request, "home.html")
        
def no_result(request):
   
    return render(request, "no_result.html")

