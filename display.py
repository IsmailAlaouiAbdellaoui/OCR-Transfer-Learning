from matplotlib import pyplot as plt
import numpy as np


def plots(img,figsize=(10,135),rows=1,interp=False,titles=None):
    if type(img[0]) is np.ndarray:
        img = np.array(img).astype(np.uint8)
        if (img.shape[-1] != 3):
            img = img.transpose((0,2,3,1))
    f = plt.figure(figsize = figsize)
    cols = len(img)//rows if len(img)%2 ==0 else len(img)//rows + 1
    for i in range(len(img)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i],fontsize = 15)
        print(np.shape(img[i]))
        plt.imshow(img[i],interpolation = None if interp else 'none')
        
def draw(imgs,labels):
    f = plt.figure()#figsize=(300/96,1000/96),dpi=96
    imgs = imgs.reshape(2,20,270)
    print("shape of imgs is : "+str(np.shape(imgs)))
    for i in range(len(imgs)):
        print(f)
        sp=f.add_subplot(i+1,1,1)
        sp.axis('Off')
        sp.set_title(labels[i])
        plt.imshow(imgs[i], interpolation=None)
    plt.subplots_adjust(bottom=0.1, top=0.9)
#    plt.subplot_tool()
    plt.show()