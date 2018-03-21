import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

def generate(count):
    labels = np.zeros((count,)).astype(int)
    images = np.zeros((count, 3, 3)).astype(int)
    k = 0
    # primary images
    for i in range(3):
        for j in range(3):
            if np.absolute(i - j) < 2:
                images[k, i, :] = [0, 255, 255]
                images[k, j, 0] = 255
                labels[k] = 0
                k += 1
                images[k,:,:] = images[k-1,:,:].T
                labels[k] = 1
                k += 1
                if i != j:
                    images[k,:,:] = np.fliplr(images[k-2,:,:])
                    labels[k] = 0
                    k += 1
                    images[k,:,:] = images[k-1,:,:].T
                    labels[k] = 1
                    k += 1
    # random noising of pixels
    for i in range(count - 22):
        j = i % 22
        images[i+22, :, :] = images[j,:,:]
        for x in range(3):
            labels[i+22] = i % 2
            for y in range(3):
                images[i+22,x,y] = np.floor(images[i+22,x,y] * np.random.uniform(.5, 1))     
    return images, labels

def flatten_save(images, labels, filename=""):
    data = np.zeros((labels.shape[0], 10)).astype(int)
    for i in range(labels.shape[0]):
        data[i, 0] = labels[i]
        data[i, 1:] = images[i,:,:].ravel()

    if len(filename) > 0:
        np.savetxt(filename, data, delimiter=",", fmt='%5d')

    return data
    
def image_save(images, labels, path):
    if os.path.exists(path):
        shutil.rmtree('%s' % path)
    os.makedirs(path)
    for i in range(labels.shape[0]):
        # if on a mac this could bork - should use os.path.join(...)
        if not os.path.exists('%s\%d' % (path, labels[i])):
            os.makedirs('%s\%d' % (path, labels[i]))
        plt.imsave('%s\%d\%08d.png' % (path, labels[i], i), images[i].astype(float), cmap='gray')

if __name__ == "__main__":
    images, labels = generate(1000)
    flatten_save(images, labels, 'fakes.csv')
    image_save(images, labels, 'data')