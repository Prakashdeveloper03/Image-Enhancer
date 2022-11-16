import cv2
import numpy as np
import streamlit as st
from rembg import remove
from PIL import Image, ImageEnhance
from scipy.interpolate import UnivariateSpline

# setting app's title, icon & layout
st.set_page_config(page_title="Image Enhancer", page_icon="🎯")

# css style to hide footer, header and main menu details
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


def LookupTable(x, y):
    spline = UnivariateSpline(
        x, y
    )  # 1-D smoothing spline fit to a given set of data points.
    return spline(range(256))  # data points ranges from 0 to 255


def cannize_image(input_image):
    rgb_img = np.array(
        input_image.convert("RGB")
    )  # Converts RGB image to numpy nd.array
    img = cv2.cvtColor(rgb_img, 1)  # Converts an image from one color space to another
    img = cv2.GaussianBlur(img, (11, 11), 0)  # Blurs an image using a Gaussian filter
    # Finds edges in an image using the Canny algorithm
    return cv2.Canny(img, 100, 150)


def sepia_effect(input_image):
    rgb_img = np.array(
        input_image.convert("RGB")
    )  # Converts RGB image to numpy nd.array
    # Sepia is one of the most commonly used filters in image editing.
    # Sepia adds a warm brown effect to photos.
    # A vintage, calm and nostalgic effect is added to images.
    img_sepia = cv2.transform(
        rgb_img,
        np.matrix(
            [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
        ),
    )
    # replaces all 255 above values to 255
    img_sepia[np.where(img_sepia > 255)] = 255
    return np.array(img_sepia, dtype=np.uint8)


def winter_effect(input_image):
    rgb_img = np.array(
        input_image.convert("RGB")
    )  # Converts RGB image to numpy nd.array
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    # In winter effect filter, The warmth of the image will be reduced.
    blue_channel, green_channel, red_channel = cv2.split(
        rgb_img
    )  # splits RGB channels separately.
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(
        np.uint8
    )  # The values in the red channel will be reduced.
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(
        np.uint8
    )  # The values in the blue channel will be increased.
    return cv2.merge((blue_channel, green_channel, red_channel))


def summer_effect(input_image):
    rgb_img = np.array(
        input_image.convert("RGB")
    )  # Converts RGB image to numpy nd.array
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    # In winter effect filter, The warmth of the image will be increased.
    blue_channel, green_channel, red_channel = cv2.split(
        rgb_img
    )  # splits RGB channels separately.
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(
        np.uint8
    )  # The values in the red channel will be increased.
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(
        np.uint8
    )  # The values in the blue channel will be reduced.
    return cv2.merge((blue_channel, green_channel, red_channel))


def sketch(input_image):
    image = np.array(input_image.convert("RGB"))  # Converts RGB image to numpy nd.array
    grey_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Converts RGB to Gray image
    invert = cv2.bitwise_not(grey_img)  # Inverts every bit of an array
    blur = cv2.GaussianBlur(
        invert, (21, 21), 0
    )  # Blurs an image using a Gaussian filter
    invertedblur = cv2.bitwise_not(
        blur
    )  # Again, Inverts every bit of an blurred image array
    return cv2.divide(
        grey_img, invertedblur, scale=256.0
    )  # Performs per-element division of two image arrays


def main():
    st.header("Image Enhancer")
    st.text("Build with Streamlit and OpenCV")
    # file uploader for getting user input image
    image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    # shows all the options list
    enhance_type = st.sidebar.selectbox(
        "Enhance Type",
        [
            "Gray Scale",
            "Pencil Effect",
            "Sepia Effect",
            "Sharp Effect",
            "Invert Effect",
            "Summer Effect",
            "Winter Effect",
            "Background Remover",
            "Brightness",
            "Blurring",
            "Contrast",
        ],
    )
    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)
        match enhance_type:
            case "Gray Scale":
                st.text("Filtered Image")
                new_img = np.array(
                    our_image.convert("RGB")
                )  # Converts RGB image to numpy nd.array
                img = cv2.cvtColor(
                    new_img, 1
                )  # Basically coloured components from the image are removed.
                gray = cv2.cvtColor(
                    img, cv2.COLOR_BGR2GRAY
                )  # cv2.cvtColor() to convert the image to greyscale.
                st.image(gray)

            case "Cannize Effect":
                st.text("Filtered image")
                result_img = cannize_image(our_image)
                st.image(result_img)

            case "Pencil Effect":
                st.text("Filtered Image")
                result_img = sketch(our_image)
                st.image(result_img)

            case "Sepia Effect":
                st.text("Filtered Image")
                result_img = sepia_effect(our_image)
                st.image(result_img)

            case "Summer Effect":
                st.text("Filtered Image")
                result_img = summer_effect(our_image)
                st.image(result_img)

            case "Winter Effect":
                st.text("Filtered Image")
                result_img = winter_effect(our_image)
                st.image(result_img)

            case "Sharp Effect":
                st.text("Filtered Image")
                new_img = np.array(our_image.convert("RGB"))
                kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
                sharpen_img = cv2.filter2D(new_img, -1, kernel)
                st.image(sharpen_img)

            case "Invert Effect":
                st.text("Filtered Image")
                rgb_img = np.array(our_image.convert("RGB"))
                # we have to do is basically invert the pixel values.
                # This can be done by subtracting the pixel values by 255
                result_img = cv2.bitwise_not(rgb_img)  # Inverts every bit of an array
                st.image(result_img)

            case "Background Remover":
                st.text("Filtered Image")
                # we have to do is basically invert the pixel values.
                # This can be done by subtracting the pixel values by 255
                result_img = remove(our_image)  # Inverts every bit of an array
                st.image(result_img)

            case "Contrast":
                st.text("Filtered Image")
                c_rate = st.sidebar.slider("Contrast", 0.5, 3.5, 3.00)
                enhancer = ImageEnhance.Contrast(our_image)
                img_output = enhancer.enhance(c_rate)
                st.image(img_output)

            case "Brightness":
                st.text("Filtered Image")
                c_rate = st.sidebar.slider("Brightness", 0.5, 3.5, 2.50)
                # we have seen filters that make the image a lot brighter, others reduce the brightness.
                # The c_rate value can be changed to get the appropriate results.
                enhancer = ImageEnhance.Brightness(our_image)
                img_output = enhancer.enhance(c_rate)
                st.image(img_output)

            case "Blurring":
                st.text("Filtered Image")
                new_img = np.array(our_image.convert("RGB"))
                blur_rate = st.sidebar.slider("Blurring", 0.5, 3.5, 1.75)
                img = cv2.cvtColor(new_img, 1)
                blur_img = cv2.GaussianBlur(img, (11, 11), blur_rate)
                st.image(blur_img)


if __name__ == "__main__":
    main()
