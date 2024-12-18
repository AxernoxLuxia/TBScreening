import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import resnet50, vgg19
from PIL import Image
import cv2

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

with tf.device('/GPU:0'):
    resnet_model = tf.keras.models.load_model('/home/axernox/Resnet50/res50_canny_1.keras', compile=False)
    vgg_model = tf.keras.models.load_model('/home/axernox/Resnet50/vgg19_canny_1.keras', compile=False)
    custom_model = tf.keras.models.load_model('/home/axernox/Resnet50/customcnn_canny.keras', compile=False)

resnet_model.compile()
vgg_model.compile()
custom_model.compile()

# Dictionary to map model selection to actual model
models = {
    "ResNet50": resnet_model,
    "VGG19": vgg_model,
    "Custom CNN": custom_model
}

# Function to preprocess the image
def clahe(image):
    image = image.convert("RGB")
    image = keras.utils.img_to_array(image)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4)
    clahe_img = clahe.apply(image.astype('uint8'))
    return clahe_img

def gabor(image):
    image = keras.utils.img_to_array(image)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = np.squeeze(image)
    ksize = 30  # Kernel size
    sigma = 1.414  # Standard deviation of the Gaussian function
    lambd = 4.0  # Wavelength of the sinusoidal factor
    gamma = 1  # Spatial aspect ratio
    psi = 0  # Phase offset
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    combined_image = np.zeros_like(image, dtype=np.float32)
    for theta in orientations:
        gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        filtered_image = cv2.filter2D(image, cv2.CV_32F, gabor_kernel)
        combined_image += filtered_image
    combined_image = cv2.normalize(combined_image, None, 0, 255, cv2.NORM_MINMAX)
    combined_image = np.uint8(combined_image)
    return combined_image

def canny(image):
    image = keras.utils.img_to_array(image)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image.astype(np.uint8)
    edges = cv2.Canny(image,100,200)
    return edges



def preprocess_image(image, target_size=(256, 256)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image = keras.utils.img_to_array(image) 
    print(image.shape)
    return np.expand_dims(image, axis=0)  

def preprocess_image1(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image = keras.utils.img_to_array(image) 
    print(image.shape)
    return np.expand_dims(image, axis=0) 

# Function to make prediction and get confidence score
def get_prediction(model, image):
    preds = model.predict(image)
    print('***********')
    print(preds)
    class_names = ['Healthy', 'Sick But No TB', 'Has Tuberculosis']
    confidence = np.max(preds) * 100
    label = class_names[np.argmax(np.round(preds,2))]
    return label, confidence

#st.title(":blue[AI-Powered TB Screening]")

#title_alignment=
#"""
#<style>
#ai-powered-tb-screening{
#    text-align:center}
#</style>
#"""
st.markdown(
    """
    <style>
    .stApp {
        background-color: #94BAD9; 
    }
    </style>
    """,
    unsafe_allow_html=True
)
#backgroundColor="f0f2f6" /* Light gray background */

st.markdown("<h1 style='text-align:center; color:#456B87;'>AI-Powered TB Screening</h1>", unsafe_allow_html=True)
st.write(
    '''
    This web application leverages advanced deep learning models, including ResNet50, VGG19, and a custom CNN, 
    to detect tuberculosis (TB) from chest X-ray images. Upload your X-ray image and choose from three models 
    to receive a diagnosis. The app provides predictions indicating whether the image suggests the presence of 
    TB or if there is no TB (Healthy) or if the person is suffering from a disease with is not TB (Sick but no TB), 
    along with the confidence score for each result.
    '''
)
st.write('Model Accuracies:')
st.write('Custom CNN: 92%')
st.write('VGG19: 91%')
st.write('ResNet50: 87%')

img_upload = st.file_uploader("Insert a PNG image", type='png')  # the only way to get images into the app is via the st.file_uploader. After an image file has been uploaded, you can process it into a format that the OpenAI API can process namely into a NumPy array.
if img_upload is not None:
    image = Image.open(img_upload)
    #img_array = np.array(image)

    st.markdown(
        """
        <style>
        .centered-image {
            display: flex;
            justify-content: center;
            
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="centered-image">', unsafe_allow_html=True)
    st.image(image, caption="Uploaded X-ray Image", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    filtered = canny(gabor(clahe(image)))
    filtered_image = Image.fromarray(filtered) 
    st.markdown('<div class="centered-image">', unsafe_allow_html=True)
    st.image(filtered_image, caption="After Filters", use_column_width = True)
    st.markdown('</div>', unsafe_allow_html=True)


    option = st.selectbox(
        "Select Model",
        ["ResNet50", "VGG19", "Custom CNN"],
    )
    st.write("Model selected: ", option)
    selected_model = models[option]
    print(selected_model)
    preprocessed_image = preprocess_image(filtered_image)
    print(preprocessed_image.shape)

    #st.button("Predict")

    st.markdown("""
        <style>
        .stButton>button {
            background-color: #CCE6FF; 
            color: black; 
            padding: 10px 24px;
            font-size: 16px;
            border-radius: 8px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #E5F2FF; 
        }
        </style>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 2, 1])

    with col2:
        if st.button("Predict"):
            if(option == "VGG19"):
                preprocessed_image = preprocess_image1(filtered_image)
            label, confidence = get_prediction(selected_model, preprocessed_image)
            st.write(f"Prediction: **{label}**") 
            st.write(f"Confidence Score: **{confidence:.2f}%**")

