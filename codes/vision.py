import cv2
import torch
import torch.nn as nn
import numpy as np

def read_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (500,500))
    return torch.from_numpy(img).permute(2,0,1)/255

def show(name, img ):
    img = img.permute(1,2,0).numpy()[::-1]

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

class cutout(nn.Module):
    def __init__(self, fill_value=0, no_holes=1, min_cut_size=100, max_cut_size=100,):
        super().__init__()
        self.fill_value = fill_value
        self.no_holes   = no_holes
        self.min_cut_size = min_cut_size
        self.max_cut_size = max_cut_size

    def forward(self, img):
        for _ in range(self.no_holes):
            box_size = torch.randint(self.min_cut_size, self.max_cut_size+1, (1,))
            x,y      = torch.randint(0,int(img.shape[-2]-box_size+1), size=(1,)), torch.randint(0,int(img.shape[-1]-box_size+1), size=(1,))
            img[:,x:x+box_size, y:y+box_size]=self.fill_value
        return img

def mixup(images, labels, alpha=1.0):
    indices = torch.randperm(len(images))
    shuff_imgs =images[indices]
    shuff_labels =labels[indices]

    lam = np.clip(np.random.beta(alpha, alpha), 0.4, 0.6)
    print(lam)

    mix_imgs = lam*images + (1-lam)*shuff_imgs
    mix_labels = lam*labels + (1-lam)*shuff_labels

    return mix_imgs, mix_labels

class cutmix(nn.Module):
    def __init__(self, min_cut_size=100, max_cut_size=100, batch_prob=0.1):
        super().__init__()
        self.min_cut_size = min_cut_size
        self.max_cut_size = max_cut_size
        self.batch_prob   = batch_prob

    def forward(self, images, labels):
        indices = torch.randperm(len(images))
        shuff_imgs =images[indices]
        shuff_labels =labels[indices]
        result_images, result_labels= [], []
        for src_img, src_trg, src_lab, tar_lab in zip(images, shuff_imgs, labels, shuff_labels):
            if torch.rand(1)>self.batch_prob:
                box_size = torch.randint(self.min_cut_size, self.max_cut_size+1, (1,))
                x,y      = torch.randint(0,int(img.shape[-2]-box_size+1), size=(1,)), torch.randint(0,int(img.shape[-1]-box_size+1), size=(1,))
                src_img[:, x:x+box_size, y:y+box_size]= src_trg[: , x:x+box_size, y:y+box_size]
                lam = (1-(box_size**2/(src_img.shape[-2]*src_img.shape[-1])))
                lab =lam*src_lab + (1. - lam)*tar_lab
                result_images.append(src_img)
                result_labels.append(lab)
        print(result_labels)
        return torch.stack(result_images), torch.stack(result_labels)



if __name__=="__main__":


    path = "../data/cat.jpg"
    save_path = "../outputs/vision/"

    img = read_image(path)
    img1 = read_image("../data/butterfly.jpg")
    data = torch.stack((img, img1), 0)
    labels = torch.tensor([[1,0],[0,1]])
    print("data shaep",data.shape, labels.shape)

    a=cutmix(min_cut_size=200, max_cut_size=200, batch_prob=0.1)
    x, y=a(data, labels)
    print(x.shape, y.shape)
    a    = torch.cat((img, x[0]),2)
    show("brightness", a)



    exit()

    data, x = mixup(data, labels)
    a    = torch.cat((img, data[1]),2)
    show("brightness", a)
    print(x)
    # show("image", img)

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
    # a    = colorjitter(brightness=2, contrast=2, saturation=2)
    # a    = a(img)
    # path = save_path+"brightness"+".png"
    # a    = torch.cat((img, a),2)
    # show("brightness", a)
    # save(path, a)

    a    = cutout(fill_value=0, no_holes=30, min_cut_size=100, max_cut_size=100)
    a    = a(img)

    a    = torch.cat((img, a),2)
    show("brightness", a)
    path = save_path+"brightness"+".png"
    # save(path, a)

    pass