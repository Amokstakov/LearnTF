import matplotlib.pyplot as plt
import numpy as np

def breaker():
    print('---------------------')
    print('---------------------')
    print('---------------------')


def __plot_image(i, predictions, true_label, imgs, class_names):
    true_label, img = true_label[i], imgs[i]

    print(img)

    breaker()
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)



    predicted_label = np.argmax(predictions)

    color = 'blue' if predicted_label == true_label else 'red'

    plt.xlabel(f"{class_names[predicted_label]}, {100*np.max(predictions)}, {class_names[true_label]}", color=color)


def __plot_value_array():
    pass

