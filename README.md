# myDataGenerator
This DataGenerator contains a generator which can read data from the txt file.


usage:
from myDataGenerator import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)
gen=test_datagen.flow_from_textfile(classes=['0', '1'],
        textfile='train.txt',
        target_size=(151, 151),
        batch_size=64,
        class_mode='binary')

