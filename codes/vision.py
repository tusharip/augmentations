import cv2
import torch
import torch.nn as nn

def read_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (500,500))
    return torch.from_numpy(img).permute(2,0,1)/255

def show(name, img ):
    img = img.permute(1,2,0).numpy()

    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save(path, img):
    img = img.permute(1,2,0).numpy()
    cv2.imwrite(path, img)

class vertical_flip(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, img):
        return img.flip(-2)

class horizontal_flip(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, img):
        return torch.flip(img, [-1])


class colorjitter(nn.Module):
    def __init__(self,brightness=0, contrast=0, saturation=0):
        super().__init__()
        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation

    def forward(self, img):
        # a= brightness(img,self.brightness)
        a = saturation(img, self.saturation)
        return a

def brightness(img, brightness):
    _range =[max(0, 1-brightness), 1+brightness]
    scale = torch.FloatTensor(1).uniform_(_range[0],_range[1])
    return img*scale

def contrast(img, contrast):
    r, g, b = img.unbind(dim=-3)
    grey = 0.2989 * r + 0.587 * g + 0.114 * b
    grey = grey.unsqueeze(dim=-3)
    mean = torch.mean(grey, dim=(-3, -2, -1), keepdim=True)

    return (contrast*img + (1-contrast) * mean).clamp(0,1)


def saturation(img, saturation):
    r, g, b = img.unbind(dim=-3)
    grey = 0.2989 * r + 0.587 * g + 0.114 * b
    grey = grey.unsqueeze(dim=-3)

    return (saturation*img + (1-saturation) * grey).clamp(0,1)

if __name__=="__main__":
    path = "../data/cat.jpg"
    save_path = "../outputs/vision/"

    img = read_image(path)
    show("image", img)

    ##############  vertical flip ######################
    # a    = vertical_flip()
    # a    = a(img)
    # path = save_path+"vertical_flip"+".png"
    # a    = torch.cat((img, a),2)
    # show("vertical_flip", a)
    # save(path, a)

    # ###############  horizontal flip ######################
    # a    = horizontal_flip(img)
    # path = save_path+"horizontal_flip"+".png"
    # a    = torch.cat((img, a),2)
    # show("horizontal_flip", a)
    # save(path, a)

###############  brightness flip ######################
    a    = colorjitter(brightness=2, contrast=2, saturation=2)
    a    = a(img)
    path = save_path+"brightness"+".png"
    a    = torch.cat((img, a),2)
    show("brightness", a)
    save(path, a)

    pass