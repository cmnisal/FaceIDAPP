import streamlit as st
import cv2

def rgb(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def show_images(images, names, num_cols, channels="RGB"):
    num_images = len(images)

    # Calculate the number of rows and columns
    num_rows = -(
        -num_images // num_cols
    )  # This also handles the case when num_images is not a multiple of num_cols

    for row in range(num_rows):
        # Create the columns
        cols = st.sidebar.columns(num_cols)

        for i, col in enumerate(cols):
            idx = row * num_cols + i

            if idx < num_images:
                img = images[idx]
                if len(names) == 0:
                    names = ["Unknown"] * len(images)
                name = names[idx]
                col.image(img, caption=name, channels=channels, width=112)


def show_faces(images, names, distances, gal_images, num_cols, channels="RGB", display=st):
    if len(images) == 0 or len(names) == 0:
        display.write("No faces detected, or gallery empty!")
        return
    # Calculate the number of rows and columns
    num_rows = -(
        -len(images) // num_cols
    )  # This also handles the case when num_images is not a multiple of num_cols

    for row in range(num_rows):
        # Create the columns
        cols = display.columns(num_cols)

        for i, col in enumerate(cols):
            idx = row * num_cols + i

            if idx < len(images):
                img = images[idx]
                name = names[idx]
                dist = distances[idx]
                col.image(img, channels=channels, width=112)
                
                if gal_images[idx] is not None:
                    col.text("  ⬍ matching ⬍")
                    col.image(gal_images[idx], caption=name, channels=channels, width=112)
                else:
                    col.markdown("")
                    col.write("No match found")
                col.markdown(
                    f"**Distance: {dist:.4f}**" if dist else f"**Distance: -**"
                )
            else:
                col.empty()
                col.markdown("")
                col.empty()
                col.markdown("")
