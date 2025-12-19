import os
import numpy as np
import tkinter as tk
from PIL import Image, ImageGrab, ImageOps
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(xtr, ytr), (xte, yte) = mnist.load_data()

xtr = xtr.reshape(xtr.shape[0], 28, 28, 1).astype("float32")
xte = xte.reshape(xte.shape[0], 28, 28, 1).astype("float32")

xtr = xtr / 255.0
xte = xte / 255.0

def make_model():
    m = Sequential()
    m.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
    m.add(MaxPooling2D((2, 2)))
    m.add(Conv2D(64, (3, 3), activation="relu"))
    m.add(MaxPooling2D((2, 2)))
    m.add(Flatten())
    m.add(Dense(128, activation="relu"))
    m.add(Dense(10, activation="softmax"))
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

path = "mnist_model.h5"

if os.path.exists(path):
    model = load_model(path)
else:
    model = make_model()
    gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    gen.fit(xtr)

    w = {}
    for i in range(10):
        w[i] = 1.0
    w[5] = 2.0
    w[6] = 2.0
    w[9] = 2.0

    model.fit(
        gen.flow(xtr, ytr, batch_size=64),
        steps_per_epoch=len(xtr) // 64,
        epochs=5,
        class_weight=w,
        validation_data=(xte, yte)
    )

    model.save(path)

root = tk.Tk()
root.title("Цифры")

c = tk.Canvas(root, width=200, height=200, bg="white")
c.pack()

lbl = tk.Label(root, text="Результат:")
lbl.pack()

px = None
py = None

def down(e):
    global px, py
    px = e.x
    py = e.y

def draw(e):
    global px, py
    if px is None:
        return
    c.create_line(px, py, e.x, e.y, width=8, fill="black", capstyle=tk.ROUND, smooth=True)
    px = e.x
    py = e.y

c.bind("<ButtonPress-1>", down)
c.bind("<B1-Motion>", draw)

def guess():
    x = root.winfo_rootx() + c.winfo_x()
    y = root.winfo_rooty() + c.winfo_y()
    x2 = x + c.winfo_width()
    y2 = y + c.winfo_height()

    img = ImageGrab.grab((x, y, x2, y2))
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)

    p = model.predict(arr)
    r = np.argmax(p)

    lbl.config(text="Результат: " + str(r))

def clear():
    c.delete("all")
    lbl.config(text="Результат:")

b1 = tk.Button(root, text="Распознать", command=guess)
b1.pack(side=tk.LEFT, padx=10, pady=10)

b2 = tk.Button(root, text="Очистить", command=clear)
b2.pack(side=tk.LEFT, padx=10, pady=10)

root.mainloop()
