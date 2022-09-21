def classifyImage(imagepath):
    import numpy as np
    from keras.models import model_from_json
    from keras.preprocessing import image
    from keras import backend as K
    K.clear_session()
    # read the json model
    file = open('Final_model.json', 'r')
    data = file.read()
    #print(data)

    file.close()

    # classifier will load the model from the data
    # data -> contents of the "my_model.json" file
    classifier = model_from_json(data)

    # load waeights
    classifier.load_weights('final_model_weights.h5')
    # load the test image
    imagetotest = image.load_img(imagepath, target_size=(256,256))
    imagetotest = image.img_to_array(imagetotest)
    imagetotest = np.expand_dims(imagetotest, axis=0)

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    result = classifier.predict(imagetotest)

    if result[0][0] == 1:
        prediction = 'Healthy'
    else:
        prediction = 'Diseased'

    return 'prediction: {}'.format(prediction)
