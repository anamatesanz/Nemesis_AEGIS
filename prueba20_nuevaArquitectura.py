# !pip3 install tensorflow==1.9.0
# !pip3 install imblearn
# !pip3 install pyrosetta

### IMPORTS

# Numpy
import numpy as np
from numpy.random import seed

# Pandas
import pandas as pd
from pandas import read_csv
import csv

# Tensorflow
import tensorflow as tf
from tensorflow import Tensor

# Sklearn
from sklearn.model_selection import train_test_split

# Keras
from keras.layers import Input, BatchNormalization, Activation, Dense, MaxPooling2D, Dropout, Flatten, Conv2D, UpSampling2D
from keras.models import Sequential, model_from_json, load_model, Model
from keras.optimizers import Adam
from keras import backend as K
import os

# Pyrosetta
from pyrosetta import init, Pose, pose_from_sequence, get_fa_scorefxn
from pyrosetta.teaching import *
init()
scorefxn = get_fa_scorefxn()

### GLOBAL VARIABLES
#Change
FIRST_TIME = True
NUM_NUCLEOTIDES = 50
BATCH_SIZE = 2
DEPTH = 32
DROPOUT = 0.4
RESULTS_CSV = "Carpeta_en_uso/resultado.csv"
DATA_BASE_CSV = "Carpeta_en_uso/prueba.csv"
POSES_FOLDER = "Carpeta_en_uso/"
EPOCHS = 4
SAMPLE_INTERVAL = 1

#No change
NUM_COL_GEN = 2 
NUM_COL_DIS = 1
NUM_CHAN_GEN = 1
NUM_CHAN_DIS = 5
NUM_ANGLES = 5
DEGREES = 360
ADENINE = [0,0]
GUANINE = [0,1]
CITOSINE = [1,0]
TYMINE = [1,1]


### INITIAL FUNCTIONS
# Creates random angles in a pose
def random_angles(pose, num):
    global DEGREES
    a = np.arange(DEGREES)
    for i in range(1, num+1):
        degrees = np.random.choice(a)
        pose.set_gamma(i, degrees)
        degrees = np.random.choice(a)
        pose.set_epsilon(i, degrees)
        degrees = np.random.choice(a)
        pose.set_delta(i, degrees)
        degrees = np.random.choice(a)
        pose.set_chi(i, degrees)
        degrees = np.random.choice(a)
        pose.set_zeta(i, degrees)
    return pose

# Estract the angles from a pose
def create_data_from_pose(pose, num):
    # print pose
    #A[ADE], G[GUA], C[CYT], T[THY]
    save_data = np.zeros(5)
    # nucleotide=pose.chain_sequence(1)[num]
    save_data[0] = pose.gamma(num+1)
    save_data[1] = pose.epsilon(num+1)
    save_data[2] = pose.chi(num+1)
    save_data[3] = pose.zeta(num+1)
    save_data[4] = pose.delta(num+1)
    return save_data

# Creates a csv with given sequences (250 angles + scoring) in a line
def create_csv_from_sequence(sequence):
    pose = pose_from_sequence(sequence)
    # pose.dump_pdb("prueba.pdb")
    poses = []
    string = []
    global NUM_NUCLEOTIDES, DATA_BASE_CSV
    for i in range(0, NUM_NUCLEOTIDES):
        string.append("Nucleoidide" + str(i) + "_Gamma")
        string.append("Nucleoidide" + str(i) + "_Epsilon")
        string.append("Nucleoidide" + str(i) + "_Chi")
        string.append("Nucleoidide" + str(i) + "_Zeta")
        string.append("Nucleoidide" + str(i) + "_Delta")
    string.append("Scoring")

    for i in range(0, 151):
        aux = []
        pose = random_angles(pose, NUM_NUCLEOTIDES)
        for i in range(0, NUM_NUCLEOTIDES):
            save_data = create_data_from_pose(pose, i)
            aux = aux + list(np.asarray(save_data))
        scoring = scorefxn(pose)
        aux.append(scoring)
        poses.append(aux)
    np.savetxt(DATA_BASE_CSV, poses, delimiter=",")
    x = read_csv(DATA_BASE_CSV)
    x.columns = string
    x.to_csv(DATA_BASE_CSV)

# Read the csv and split the data into X(sequences) and y (scoring)
def read_data_and_split_into_X_and_y():
    # df.head()
    # df.describe() #Count, mean, std,min....
    df = pd.read_csv(DATA_BASE_CSV)
    df.head()
    #df = df.drop("Time",axis=1)
    aux = df.drop(df.columns[[0, (5*50)+1]], axis=1).values
    y = df["Scoring"].values
    # maximum=int(50/5)
    # X=np.zeros(shape=(len(aux),50,5))
    # for i in range(0,len(aux)):
    #    row=aux[i,:]
    #    X[i] = [ row [i:i + 5] for i in range(0, len(row), 5) ]
    # print(X)
    # print(y)
    X = aux
    return X, y


create_csv_from_sequence("A[ADE]"* NUM_NUCLEOTIDES)
X, y = read_data_and_split_into_X_and_y()

#With the X and y construct 2 groups= train and test groups
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

os.environ["KERAS_BACKEND"] = "tensorflow"  # Set Keras-Tensorflow environment
K.set_image_dim_ordering("th")
# The results are a little better when the dimensionality of the random vector is only 10.
randomDim = 100
# The dimensionality has been left at 100 for consistency with other GAN implementations.

### GENERATOR
# Build the generator model
def build_my_generator(data_col=NUM_COL_GEN, data_row=NUM_NUCLEOTIDES, channels=NUM_CHAN_GEN, batch_size=BATCH_SIZE, angles=NUM_ANGLES):

    global DEPTH, DROPOUT

    model = Sequential()

    model.add(Conv2D(DEPTH, 5, data_format="channels_first", padding="same"),)
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(DROPOUT))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(int(DEPTH*2), 5, padding="same"))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(DROPOUT))
    model.add(Activation("relu"))

    model.add(Conv2D(int(DEPTH*4), 5, padding="same"))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(DROPOUT))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(5, 1)))

    model.add(Conv2D(int(DEPTH*8), 5, padding="same"))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(DROPOUT))
    model.add(Activation("relu"))

    model.add(UpSampling2D(size=((5, 1))))
    model.add(Conv2D(int(DEPTH*4), 5, padding="same"))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(DROPOUT))
    model.add(Activation("relu"))

    model.add(Conv2D(int(DEPTH*2), 5, padding="same"))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(DROPOUT))
    model.add(Activation("relu"))

    model.add(UpSampling2D(size=((2, 1))))
    model.add(Conv2D(int(DEPTH), 5, padding="same"))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(DROPOUT))
    model.add(Activation("relu"))

    model.add(Conv2D(angles, 5, padding="same"))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(DROPOUT))
    model.add(Activation("relu"))

    base_pairs = Input(shape=(channels, data_row, data_col))
    degrees = model(base_pairs)

    # sumed = Concatenate(axis=1)([base_pairs, degrees]) #Add base pairs to output
    #G = Model(base_pairs, sumed)

    G = Model(base_pairs, degrees)
    G.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-3))

    return G
# Constructs the generator
my_generator = build_my_generator()
my_generator.summary()
## From (2, 1, 50, 2) to (2, 5, 50, 1)

# a=np.arange(360)
# noise=np.zeros([2,1,50,1])
# for i in range (0,2):
#     for j in range (0,1):
#         for w in range (0,50):
#             for y in range (0,1):
#                 #noise[i,j,w,y]= np.random.random()
#                 noise[i,j,w,y]= np.random.choice(a)

# Create random nucleotides (A,G,C,T)
def random_nucleotide():
    value = np.random.choice((1, 2, 3, 4))
    global ADENINE, GUANINE,CITOSINE,TYMINE
    if value == 1:
        return ADENINE
    elif value == 2:
        return GUANINE
    elif value == 3:
        return CITOSINE
    else:
        return TYMINE

# Create sequences (#batch_size) from random nucleotides
def noise_nucleotides(batch_size):
    global NUM_NUCLEOTIDES
    noise = np.zeros([batch_size, 1, NUM_NUCLEOTIDES, 2])
    for i in range(0, batch_size):
        for y in range(0, 1):
            for w in range(0, NUM_NUCLEOTIDES):
                #print (random_nucleotide())
                noise[i, y, w] = random_nucleotide()
    return noise

noise=noise_nucleotides(BATCH_SIZE)
# print(noise)
#print(noise.shape) 
# (2, 1, 50, 2)
first_generated_data = my_generator.predict(noise)
#print("First generated data: %s" %first_generated_data)
#print(first_generated_data.shape)
# (2, 5, 50, 1)

### DISCRIMINATOR
# Creates the discriminator model
def build_my_discriminator(data_col=NUM_COL_DIS, data_row=NUM_NUCLEOTIDES, channels=NUM_CHAN_DIS, batch_size=BATCH_SIZE, angles=NUM_ANGLES):
    # channels=6 #Add base pairs to output

    global DEPTH, DROPOUT

    model = Sequential()

    model.add(Conv2D(DEPTH, 5, data_format="channels_first", padding="same"))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(DROPOUT))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(DEPTH*2, 5, padding="same"))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(DROPOUT))
    model.add(Activation("relu"))

    model.add(Conv2D(DEPTH*4, 5, padding="same"))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(DROPOUT))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(5, 1)))
    model.add(Conv2D(DEPTH*8, 5, padding="same"))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(DROPOUT))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(5, 1)))
    model.add(Conv2D(DEPTH*16, 5, padding="same"))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dropout(DROPOUT))
    model.add(Activation("relu"))

    model.add(Flatten())
    model.add(Dense(DEPTH*8))
    model.add(Activation("relu"))
    model.add(Dropout(DROPOUT))
    model.add(Dense(DEPTH*4))
    model.add(Activation("relu"))
    model.add(Dropout(DROPOUT))
    model.add(Dense(DEPTH*2))
    model.add(Activation("relu"))
    model.add(Dropout(DROPOUT))
    model.add(Dense(DEPTH))
    model.add(Activation("relu"))
    model.add(Dropout(DROPOUT))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    img = Input(shape=(channels, data_row, data_col))
    decision = model(img)
    D = Model(img, decision)
    D.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-3))
    return D

# Constructs the discriminator
my_discriminator = build_my_discriminator()
my_discriminator.summary()
## From (2, 5, 50, 1) to (2, 1)

first_prediction = my_discriminator.predict(first_generated_data)
# print(first_prediction.shape)
# print(first_prediction)

### GAN
# The discriminator is freeze in GNA model
def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

# Creates the GAN model
def build_mygan(generator, discriminator, data_col=NUM_COL_GEN, data_row=NUM_NUCLEOTIDES, channels=NUM_CHAN_GEN):
    set_trainability(discriminator, False)
    GAN_in = Input(shape=(channels, data_row, data_col))
    x = generator(GAN_in)
    GAN_out = discriminator(x)
    GAN = Model(GAN_in, GAN_out)
    GAN.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-3))
    return GAN

#Constructs the GAN model
my_gan = build_mygan(generator=my_generator, discriminator=my_discriminator)
my_gan.summary()
## From (2, 1, 50, 2) to (2, 1)

#print(noise.shape)
first_gan_data = my_gan.predict(noise)
#print("First GAN data: %s" %first_gan_data)
#print(first_gan_data.shape)

### SAVE PDBs AND CSVs
# Save the given sequence in a csv = 1 line contain: Sequence, Degrees and Score
def save_to_csv(seq, x, score):
    headers = ("Sequence", "Degrees", "Score")
    table = [seq, x.ravel(), score]
    global FIRST_TIME, RESULTS_CSV
    if (FIRST_TIME == True):
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerow(table)
        FIRST_TIME = False
    else:
        with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(table)

# Save a pose in a pbd format
def create_pdb_and_save(pose, score, epoch):
    global POSES_FOLDER
    pose.dump_pdb(POSES_FOLDER+"pose_" +
                  str(epoch)+"_Scoring_"+str(score)+".pdb")

# Save the predicted figures (each x epochs) in a csv (per lines) 
def save_created_figures(epoch):
    noise = noise_nucleotides(1)
    gen_img = my_generator.predict(noise)
    pose_new = Pose()
    sequence = ""
    global NUM_NUCLEOTIDES,ADENINE,CITOSINE,GUANINE,TYMINE

    for no in range(0, NUM_NUCLEOTIDES):
        if sum(noise[0, 0, no] == ADENINE) == 2:
            sequence = sequence+"A[ADE]"
        elif sum(noise[0, 0, no] == CITOSINE) == 2:
            sequence = sequence+"C[CYT]"
        elif sum(noise[0, 0, no] == GUANINE) == 2:
            sequence = sequence+"G[GUA]"
        elif sum(noise[0, 0, no] == TYMINE) == 2:
            sequence = sequence+"T[THY]"
    pose_new = pose_from_sequence(sequence)

    for k in range(0, NUM_NUCLEOTIDES):
        pose_new.set_gamma(k+1, gen_img[0, 0, k])
        pose_new.set_epsilon(k+1, gen_img[0, 1, k])
        pose_new.set_delta(k+1, gen_img[0, 2, k])
        pose_new.set_chi(k+1, gen_img[0, 3, k])
        pose_new.set_zeta(k+1, gen_img[0, 4, k])
    score = scorefxn(pose_new)
    save_to_csv(sequence, gen_img[0, 0], score)
    create_pdb_and_save(pose_new, score, epoch)

### TRAIN
# Train the model
def train(X_train, epochs=EPOCHS, batch_size=BATCH_SIZE, sample_interval=SAMPLE_INTERVAL):
    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
    # Discriminator
    # Select a random batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        #noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        noise = noise_nucleotides(batch_size)
        #print (noise.shape)

    # Generate a batch of new images
        gen_imgs = my_generator.predict(noise)
        print(gen_imgs.shape)

    # Train the discriminator
        d_loss_real = my_discriminator.train_on_batch(imgs, valid)
        d_loss_fake = my_discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Generator
        noise = noise_nucleotides(batch_size)
    # Train the generator (to have the discriminator label samples as valid)
        g_loss = my_gan.train_on_batch(noise, valid)

    # Plot the progress
        print("Epoch %d [D loss: %f] [G loss: %f]" % (epoch, d_loss, g_loss))
    # If at save interval => save generated image samples
        if epoch % sample_interval == 0:
            save_created_figures(epoch)


# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
train(X_train=first_generated_data)
# new_gan=my_gan.predict(noise)
# print(new_gan)

### SAVE MODELS AND ITS WEIGHTS
# Freeze models in order to save them
def freeze(model):
    for layer in model.layers:
        layer.trainable = False

        if isinstance(layer, Model):
            freeze(layer)
freeze(my_gan)
freeze(my_discriminator)
freeze(my_generator)

# Serialize model to JSON and save the models
my_gan_json = my_gan.to_json()
with open(POSES_FOLDER+"gan.json", "w") as json_file:
    json_file.write(my_gan_json)
my_generator_json = my_generator.to_json()
with open(POSES_FOLDER+"generator.json", "w") as json_file:
    json_file.write(my_generator_json)
my_discriminator_json = my_discriminator.to_json()
with open(POSES_FOLDER+"discriminator.json", "w") as json_file:
    json_file.write(my_discriminator_json)

# Serialize weights to HDF5 and save them
my_generator.save_weights(POSES_FOLDER+"generator_weights.h5")
my_discriminator.save_weights(POSES_FOLDER+"discriminator_weights.h5")
my_gan.save_weights(POSES_FOLDER+"gan_weights.h5")

### OPEN MODELS AND THEIR WEIGHTS 
# Load JSON and create model
json_file = open(POSES_FOLDER+"generator.json", "r")
loaded_generator = json_file.read()
json_file.close()
my_generator_new = model_from_json(loaded_generator)
json_file = open(POSES_FOLDER+"discriminator.json", "r")
loaded_discriminator = json_file.read()
json_file.close()
my_discriminator_new = model_from_json(loaded_discriminator)
json_file = open(POSES_FOLDER+"gan.json", "r")
loaded_gan = json_file.read()
json_file.close()
my_gan_new = model_from_json(loaded_gan)

# Load weights into the new models
my_discriminator_new.load_weights(POSES_FOLDER+"discriminator_weights.h5")
my_generator_new.load_weights(POSES_FOLDER+"generator_weights.h5")
my_gan_new.load_weights(POSES_FOLDER+"gan_weights.h5")

# Evaluate loaded models on test data
my_gan_new.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-3))
noise=noise_nucleotides(1)
new_gan = my_gan_new.predict(noise)
print(new_gan)
my_generator_new.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-3))
new_value = my_generator_new.predict(noise)
print(new_value.shape)
my_discriminator_new.compile(
    loss="binary_crossentropy", optimizer=Adam(lr=1e-3))
new_pred = my_discriminator_new.predict(new_value)
print(new_pred)