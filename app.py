import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.title('Sports person classfier')

image_messi = Image.open('/home/suhailpk/projects/classification_images/server/dp/Leo Messi 🔟 (_WeAreMessi)   Twitter.jpg')
image_kohli = Image.open('/home/suhailpk/projects/classification_images/server/dp/0b43460429.jpg')
image_maria = Image.open('/home/suhailpk/projects/classification_images/server/dp/Maria Sharapova HD Latest Wallpapers (1).jpg')
image_roger = Image.open('/home/suhailpk/projects/classification_images/server/dp/1_478 Roger Federer Photos - Free (1).jpg')
image_serena = Image.open('/home/suhailpk/projects/classification_images/server/dp/09Hunter1-superJumbo.jpg')
#st.image(image_messi, caption='messi',width=250)
#st.image(image_kohli, caption='messi',width=250)

col1,col2,col3,col4,col5 = st.columns(5)
col1.image(image_messi, caption='messi',width=120)
col2.image(image_kohli, caption='messi',width=120)
col3.image(image_maria, caption='messi',width=120)
col4.image(image_roger, caption='messi',width=120)
col5.image(image_serena, caption='messi',width=120)

uploaded_file = st.file_uploader("Choose a file")
model = load_model('/home/suhailpk/projects/classification_images/server/model/model_2.h5')
class_names = ['kohli', 'maria sharapova', 'messi', 'roger federer', 'serena williams']
def prediction(target_file):
    test_images = image.load_img(target_file,target_size=(256,256))
    test_images = image.img_to_array(test_images)
    test_images = np.expand_dims(test_images,axis=0)
    return class_names[np.argmax(model.predict(test_images))]

if uploaded_file is not None:
    st.write(prediction(uploaded_file))