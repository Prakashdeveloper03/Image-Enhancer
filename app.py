import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance
from scipy.interpolate import UnivariateSpline


def LookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))


def cannize_image(our_image):
    new_img = np.array(our_image.convert("RGB"))
    img = cv2.cvtColor(new_img, 1)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    return cv2.Canny(img, 100, 150)


def sepia_effect(our_image):
    img_sepia = np.array(our_image.convert("RGB"))
    img_sepia = cv2.transform(
        img_sepia,
        np.matrix(
            [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
        ),
    )
    img_sepia[np.where(img_sepia > 255)] = 255
    img_sepia = np.array(img_sepia, dtype=np.uint8)
    return img_sepia


def winter_effect(our_image):
    img = np.array(our_image.convert("RGB"))
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel, red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    return cv2.merge((blue_channel, green_channel, red_channel))


def summer_effect(our_image):
    img = np.array(our_image.convert("RGB"))
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel, red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    return cv2.merge((blue_channel, green_channel, red_channel))


def sketch(our_image):
    image = np.array(our_image.convert("RGB"))
    grey_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    invert = cv2.bitwise_not(grey_img)
    blur = cv2.GaussianBlur(invert, (21, 21), 0)
    invertedblur = cv2.bitwise_not(blur)
    return cv2.divide(grey_img, invertedblur, scale=256.0)


def main():
    st.header("Image Enhancer")
    st.text("Build with Streamlit and OpenCV")
    image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    enhance_type = st.sidebar.selectbox(
        "Enhance Type",
        [
            "Gray Scale",
            "Pencil Effect",
            "Sepia Effect",
            "Invert Effect",
            "Summer Effect",
            "Winter Effect",
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
                new_img = np.array(our_image.convert("RGB"))
                img = cv2.cvtColor(new_img, 1)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

            case "Invert Effect":
                st.text("Filtered Image")
                image = np.array(our_image.convert("RGB"))
                result_img = cv2.bitwise_not(image)
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
