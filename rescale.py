from PIL import  Image
import  os
import argparse

#cutom rescale the image

def rescale_imagess(directory,size):
    for img in os.listdir(directory):
        im = Image.open(directory+img)
        im_resized = im.resize(size,Image.ANTIALIAS)
        im_resized.save(directory+img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Recale images")
    parser.add_argument('-d','--directory',type =str,required=True,help='Directory containing the images')
    parser.add_argument('-s','--size',type=int,nargs=2,required=True,metavar=('width','height'),help='Image size')
    args = parser.parse_args()
    rescale_imagess(args.directory,args.size)