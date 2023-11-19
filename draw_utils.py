
import matplotlib.pyplot as plt

def show_images(images, titles, size=5):
    
    count = len(images)
    plt.figure(figsize=(count * size, size))
    
    for i in range(count):    
        plt.subplot(1, count, i + 1)
        plt.title(titles[i])
        plt.imshow(images[i])
        # plt.grid(False)
        plt.axis('off')

        # if i == (count - 1):
        #     plt.colorbar()