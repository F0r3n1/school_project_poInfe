#перед запуском кода ввести в консоль pip install numpy pillow tensorflow

import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images /255
test_images = test_images /255
#создание модели на основе Sequential
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=32)
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

#для создания окна основного
root = tk.Tk()
root.title("Проект")
root.resizable(False, False)

canvas_width = 280
canvas_height = 280
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')
canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

image = Image.new("L", (canvas_width, canvas_height), 'white')
draw = ImageDraw.Draw(image)
#для рисования
def paint(event):
    x1, y1 = (event.x - 16), (event.y - 16)
    x2, y2 = (event.x + 16), (event.y + 16)
    canvas.create_oval(x1, y1, x2, y2, fill='black', width=0)
    draw.ellipse([x1, y1, x2, y2], fill='black')

canvas.bind("<B1-Motion>", paint)
#предикт
def predict_digit():
    img = image.resize((28,28))
    img_array = np.array(img)
    img_array = 255 - img_array
    img_array = img_array /255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)
    result_label.config(text= "Сеть думает, что это: " + str(digit) )
#очистить канвас
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0,0,canvas_width,canvas_height], fill='white')
    result_label.config(text= "")

predict_button = tk.Button(root, text="распознать цифру", command=predict_digit )
predict_button.grid(row=1, column=0, pady=5, padx=5, sticky="ew")

clear_button = tk.Button(root, text="очистить окно рисования", command=clear_canvas )
clear_button.grid(row=1, column=1, pady=5, padx=5, sticky="ew")

result_label = tk.Label(root, text= "" )
result_label.grid(row=2, column=0, columnspan=2)

root.mainloop()
