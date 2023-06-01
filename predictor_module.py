import tensorflow as tf
import numpy as np

def predictor(predictor_model, location:str, img_height:int = 240, img_width:int = 320):
    # Load the image.
    img = tf.keras.utils.load_img(
       location, target_size=(img_height, img_width)
    )

    # Convert the image to an array.
    # Create a batch of size 1.
    img_array = tf.expand_dims(tf.keras.utils.img_to_array(img),0)

    # Get the predictions from the model.
    predictions = predictor_model.predict(img_array)

    # Get the scores for the predictions.
    score = tf.nn.softmax(predictions[0])

    # Sort the predictions by score.
    sorted_predictions = np.argsort(score)[::-1]

    # Take the predictions.
    top_predictions = sorted_predictions[:]

    # class names.
    class_names = ['negative','positive']
    
    
    # Print the predictions.
    for i in range(len(top_predictions)):
        print(
            "The image most likely belongs to {} with a {:.4f} percent confidence."
            .format(class_names[top_predictions[i]], 100 * score[top_predictions[i]])
        )