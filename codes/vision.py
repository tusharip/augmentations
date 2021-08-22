import cv2
import torch

def read_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (500,500))
    return torch.from_numpy(img).permute(2,0,1)

def show(name, img ):
    img = img.permute(1,2,0).numpy()

    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save(path, img):
    img = img.permute(1,2,0).numpy()
    cv2.imwrite(path, img)

def vertical_flip(img):
    return img.flip(-2)

def horizontal_flip(img):
    return torch.flip(img, [-1])






if __name__=="__main__":
    path = "../data/cat.jpg"
    save_path = "../outputs/vision/"

    img = read_image(path)
    show("image", img)

    ###############  vertical flip ######################
    a    = vertical_flip(img)
    path = save_path+"vertical_flip"+".png"
    a    = torch.cat((img, a),2)
    show("vertical_flip", a)
    save(path, a)

    ###############  horizontal flip ######################
    a    = horizontal_flip(img)
    path = save_path+"horizontal_flip"+".png"
    a    = torch.cat((img, a),2)
    show("horizontal_flip", a)
    save(path, a)


    pass