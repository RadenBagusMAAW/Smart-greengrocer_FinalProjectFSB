from io import BytesIO
from PIL import Image
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
import tensorflow as tf

def main():
    model= tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=64,activation="relu",kernel_size=3,input_shape=(150,150,3)))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(tf.keras.layers.Conv2D(filters=64, activation="relu",kernel_size=3))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(tf.keras.layers.Conv2D(filters=128, activation="relu",kernel_size=3))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(tf.keras.layers.Conv2D(filters=128, activation="relu",kernel_size=3))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=512, activation="relu"))
    model.add(tf.keras.layers.Dense(units=15,activation="softmax"))

    model.compile(
        optimizer = 'Adam', 
        loss = 'categorical_crossentropy', 
        metrics = ['accuracy']
    )

    model.load_weights('sayur_weightsfix.h5') 

    def predict_image(image_upload, model = model):
        im = Image.open(image_upload)
        resized_im = im.resize((150, 150))
        im_array = np.asarray(resized_im)
        im_array = im_array*(1/225)
        im_input = tf.reshape(im_array, shape = [1, 150, 150, 3])

        predict_array = model.predict(im_input)[0]
        bean_proba = predict_array[0]
        bitterGourd_proba = predict_array[1]
        bottleGourd_proba = predict_array[2]
        brinjal_proba = predict_array[3]
        broccoli_proba = predict_array[4]
        cabbage_proba = predict_array[5]
        capsicum_proba = predict_array[6]
        carrot_proba = predict_array[7]
        cauliflower_proba = predict_array[8]
        cucumber_proba = predict_array[9]
        papaya_proba = predict_array[10]
        potato_proba = predict_array[11]
        pumpkin_proba = predict_array[12]
        radish_proba = predict_array[13]
        tomato_proba = predict_array[14]

        s = [bean_proba, bitterGourd_proba, bottleGourd_proba, brinjal_proba, broccoli_proba,
             cabbage_proba,capsicum_proba,carrot_proba,cauliflower_proba,cucumber_proba,
             papaya_proba, potato_proba, pumpkin_proba, radish_proba, tomato_proba]

        import pandas as pd
        df = pd.DataFrame(predict_array)
        df = df.rename({0:'Probability'}, axis = 'columns')
        prod = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd',"Brinjal",
            "Broccoli","Cabbage","Capsicum","Carrot","Cauliflower",
           "Cucumber","Papaya","Potato","Pumpkin","Radish","Tomato"]
        df['Product'] = prod
        df = df[['Product', 'Probability']]

        predict_label = np.argmax(model.predict(im_input))

        if predict_label == 0:
            predict_product = 'Bean'
        elif predict_label == 1:
            predict_product = 'Bitter_Gourd'
        elif predict_label == 2:
            predict_product = 'Bottle_Gourd'
        elif predict_label == 3:
            predict_product = 'Brinjal'
        elif predict_label == 4:
            predict_product = 'Broccoli'
        elif predict_label == 5:
            predict_product = 'Cabbage'
        elif predict_label == 6:
            predict_product = 'Capsicum'
        elif predict_label == 7:
            predict_product = 'Carrot'
        elif predict_label == 8:
            predict_product = 'Cauliflower' 
        elif predict_label == 9:
            predict_product = 'Cucumber'
        elif predict_label == 10:
            predict_product = 'Papaya'
        elif predict_label == 11:
            predict_product = 'Potato'
        elif predict_label == 12:
            predict_product = 'Pumpkin'
        elif predict_label == 13:
            predict_product = 'Radish'
        elif predict_label == 14:
            predict_product = 'Tomato'

        return predict_product, df, im, s
    
    st.set_page_config(page_title=' Smart Greengrocer 🥒', page_icon = 'Iconicon-Veggies-Tomato.ico')
    
    st.sidebar.header('Please Enter Image Link')
    st.sidebar.markdown(
        " \
        Example link for <a href='https://m.media-amazon.com/images/I/71-ZJyXbZQL._SL1417_.jpg' style='text-decoration: none;'>Vegetable1</a>, \
                         <a href='https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Potato_heart_mutation.jpg/878px-Potato_heart_mutation.jpg' style='text-decoration: none;'>Vegetable2</a>, \
                         <a href='https://assets.pikiran-rakyat.com/crop/0x0:0x0/750x500/photo/2021/07/17/582932815.jpg' style='text-decoration: none;'>Vegetable3</a>", \
        unsafe_allow_html=True
    )
    text = st.sidebar.markdown("The model currently can predict 15 variants of vegetables such as Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli, Cabbage, Capsicum, Carrot, Cauliflower, Cucumber, Papaya, Potato, Pumpkin, Radish and Tomato")
    image_url = st.sidebar.text_input("Please use JPG or JPEG image for better prediction")
    

    if image_url == "":
        st.sidebar.image('https://media.giphy.com/media/QWvra259h4LCvdJnxP/giphy.gif', width=300)
        st.markdown("<h1 style='text-align: center;'>🍅 Smart Greengrocer: Vegetable classification from Images 🥒</h1>", unsafe_allow_html=True)
        st.markdown("""
                    """)
        st.image('png-clipart-assorted-vegetable-lot-vegetable-fruit-basket-century-farms-international-fruits-and-vegetables-natural-foods-leaf-vegetable.png', width=700)
        st.markdown("<h3 style='text-align: center;'>Project by <a href='linkedin.com/in/adryanpaw' style='text-decoration: none; color:white;'>Raden Bagus M A A W</a></h3>", unsafe_allow_html=True)

    else:
        try:
            file = BytesIO(urlopen(image_url).read())
            img = file
            label, df_output, uploaded_image, s = predict_image(img)
            st.sidebar.image(uploaded_image, width = None)

            st.markdown("<h1 style='text-align: center;'>The Image is Detected as {}</h1>".format(label), unsafe_allow_html=True)
            st.markdown("""
                        """)
            
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(12,6))
            ax = sns.barplot(x = 'Product', y = 'Probability', data = df_output)
            plt.xticks(rotation=60)
            plt.xlabel('')

            for i,p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x()+p.get_width()/2.,
                    height + 0.01, str(round(s[i]*100,2))+'%',
                    ha="center") 

            st.pyplot(fig)
        except:
            st.sidebar.image('emot.gif')
            st.markdown("<h1 style='text-align: center; color:red;'>Oh, No! 😱</h1>", unsafe_allow_html=True)
            st.markdown("""
                        """)
            st.image('png-clipart-assorted-vegetable-lot-vegetable-fruit-basket-century-farms-international-fruits-and-vegetables-natural-foods-leaf-vegetable.png', width=700)
            st.markdown("<h2 style='text-align: center;'>Please Use Another Link Image 🙏🏻</h2>", unsafe_allow_html=True)
        
if __name__ == '__main__':
    main()