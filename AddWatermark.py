import cv2

def AddWatermark(img,logo,logo_size,logo_position):
    w_logo, h_logo = logo_size
    x_logo, y_logo = logo_position
    logo_resize = cv2.resize(logo,(w_logo,h_logo))

    # crop :
    crop_img = img[x_logo:w_logo,y_logo:h_logo]
    result = cv2.addWeighted(crop_img,1,logo_resize,0.3,0)

    img[x_logo:w_logo,y_logo:h_logo] = result
    return img