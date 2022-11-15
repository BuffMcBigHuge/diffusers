from PIL import Image, ExifTags
import os
from os import listdir
import sys
import glob

from resizeimage import resizeimage

# read every file in the directory './data/
# take in argument for directory
if len(sys.argv) > 1:
    directory = sys.argv[1]
else:
    directory = './data/dog'

final_size = 512;

for file in listdir(directory):
    print(file)
    if (not(file.endswith('_resized.jpg')) and (file.endswith('.jpg') or file.endswith('.JPG') or file.endswith('.JPEG') or file.endswith('.jpeg'))):
    # read the file
        image = Image.open(directory + '/' + file)

        # check rotation
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation]=='Orientation':
                    break
            exif=dict(image._getexif().items())

            if exif[orientation] == 3:
                image=image.transpose(Image.ROTATE_180)
            elif exif[orientation] == 6:
                image=image.transpose(Image.ROTATE_270)
            elif exif[orientation] == 8:
                image=image.transpose(Image.ROTATE_90)
        except (AttributeError, KeyError, IndexError):
            # cases: image don't have getexif
            pass

        # resize cover
        cover = resizeimage.resize_cover(image, [512, 512])
        # save the file
        cover.save(directory + '/' + file + '_resized.jpg', image.format)
        # remove the original file
        os.remove(directory + '/' + file)
