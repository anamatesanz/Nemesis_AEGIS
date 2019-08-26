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
from keras.layers import Input, BatchNormalization, Activation, Dense, MaxPooling2D, Dropout, Flatten, Conv2D, UpSampling2D, concatenate, Conv2DTranspose
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
BATCH_SIZE = 100
DROPOUT = 0.9
RESULTS_CSV = "Carpeta_en_uso/resultado.csv"
DATA_BASE_CSV = "Carpeta_en_uso/prueba.csv"
POSES_FOLDER = "Carpeta_en_uso/"
EPOCHS = 1000
SAMPLE_INTERVAL = 100

#No change
NUM_COL_GEN = 1
NUM_COL_DIS = 1
NUM_CHAN_GEN = 4
NUM_CHAN_DIS = 5
NUM_ANGLES = 5
TYPE_NUCLEOTIDES = 4
DEPTH = NUM_ANGLES*TYPE_NUCLEOTIDES #20
NUM_VALUES_DB=None


# CHANNELS ORDER:
# Degrees: gamma, epsilon, delta, chi and zeta
# Nucleiotides: A, G, C and T

### INITIAL FUNCTIONS
# Read the csv and split the data into X(sequences) and y (scoring)
def read_data_and_split_into_Degrees_Sequence_and_Scoring():
    global DATA_BASE_CSV
    df = pd.read_csv(DATA_BASE_CSV)
    df.head()
    deg = df["Degrees"].values
    sco = df["Score"].values
    seq = df["Sequence"].values
    return seq, deg,sco

sequence,degrees,scoring = read_data_and_split_into_Degrees_Sequence_and_Scoring()
NUM_VALUES_DB=len(degrees)

def reshape_degrees(degrees):
    global NUM_NUCLEOTIDES, NUM_ANGLES,NUM_VALUES_DB
    new_array=[]
    for i in range(0,NUM_VALUES_DB):
        sentence =degrees[i]
        aux = sentence.split(',')
        aux[0] = aux[0].replace("["," ")
        aux[249] = aux[249].replace("]"," ")
        my_array=[]
        for item in aux:
            my_array.append(float(item))
        gamma=[]
        epsilon=[]
        delta=[]
        chi=[]
        zeta=[]
        j=0
        while j<((NUM_ANGLES*NUM_NUCLEOTIDES)-1):
            gamma.append(my_array[j])
            j=j+1
            epsilon.append(my_array[j])
            j=j+1
            delta.append(my_array[j])
            j=j+1
            chi.append(my_array[j])
            j=j+1
            zeta.append(my_array[j])
            j=j+1
        new_array.append([gamma,epsilon,delta,chi,zeta])
    reshape_array=np.asarray(new_array)
    return reshape_array.reshape((NUM_VALUES_DB,NUM_ANGLES,NUM_NUCLEOTIDES,1))

def reshape_sequence(sequence):
    global NUM_NUCLEOTIDES,NUM_VALUES_DB,TYPE_NUCLEOTIDES
    new_array=[]
    for i in range(0,NUM_VALUES_DB):
        sentence =sequence[i]
        sentence=sentence.split("]")
        aux=np.asarray(sentence)
        noise=np.zeros([TYPE_NUCLEOTIDES,NUM_NUCLEOTIDES])
        j=0
        while j<(NUM_NUCLEOTIDES):
            if aux[j]=="A[ADE":
                noise[0,j]=1
            elif aux[j]=="G[GUA":
                noise[1,j]=1
            elif aux[j]=="C[CYT":
                noise[2,j]=1
            elif aux[j]=="T[THY":
                noise[3,j]=1
            j=j+1
        new_array.append(noise)
    reshape_array=np.asarray(new_array)
    return reshape_array.reshape((NUM_VALUES_DB,TYPE_NUCLEOTIDES,NUM_NUCLEOTIDES,1))


degrees_reshaped=reshape_degrees(degrees)
#print(degrees_reshaped.shape) = (NUM_VALUES_DB,5,50,1)
sequence_reshaped=reshape_sequence(sequence)
#print(sequence_reshaped.shape) = (NUM_VALUES_DB,4,50,1)

#With the X and y construct 2 groups= train and test groups
X_train, X_test, Y_train, Y_test = train_test_split(sequence_reshaped,degrees_reshaped, test_size=0.1)

os.environ["KERAS_BACKEND"] = "tensorflow"  # Set Keras-Tensorflow environment
K.set_image_dim_ordering("th") #(channels, rows, cols)

### GENERATOR
# Build the generator model
def build_my_generator(data_col=NUM_COL_GEN, data_row=NUM_NUCLEOTIDES, channels=NUM_CHAN_GEN, batch_size=BATCH_SIZE, angles=NUM_ANGLES):

    global DEPTH, DROPOUT
    # Construct the model
    # Input -> (4,50,1)
    base_pairs = Input(shape=(channels, data_row, data_col))
    # Applies the model to the data:
    
    # CNN with diabolo form: 2 parts -> convolution and deconvolution
    # Convolution part:
    # Convolution layer
    
    x = Conv2D(DEPTH, (5,1), padding="same")(base_pairs)
    # Conv2 : 1. number of filters,
    # 2.shape of filters 
    # 3."same"-> results in padding the input such that the output has the same length as the original input.
    x = Dropout(DROPOUT)(x)
    # Dropuot add variance (Reduce the overfitting)
    x = Activation("relu")(x)
    # Relu: rectifier function
    # Output -> (20, 50, 1)

    # Pooling layer:
    # Reduce the size of the images as much as possible: it reduces the complexity of the model without reducing it’s performance.
    x = MaxPooling2D(pool_size=(2, 1))(x)
    # Divide rows and columns by pool_size
    # Output => (20, 25, 1)

    x = Conv2D((DEPTH*2), (5,1), padding="same")(x)
    x = Dropout(DROPOUT)(x)
    x = Activation("relu")(x)
    # Output => (40, 25, 1)
    
    x = MaxPooling2D(pool_size=(5, 1))(x)
    # Output => (40, 5, 1)

    x = Conv2D((DEPTH*4), (5,1), padding="same")(x)
    x = Dropout(DROPOUT)(x)
    x = Activation("relu")(x)
    # Output => (80, 5, 1)
    
    x = MaxPooling2D(pool_size=(5, 1))(x)
    # Output => (80, 1, 1)

    # Deconvolution part:
    # Unpooling layer (pooling adversary)
    x = UpSampling2D(size=(5, 1))(x)
    # Output => (80, 5, 1)

    # Deconvolution layer
    x = Conv2DTranspose(DEPTH*2, (5,1), padding="same")(x)
    x = Dropout(DROPOUT)(x)
    x = Activation("relu")(x)
    # Output => (40, 5, 1)

    x = UpSampling2D(size=(5, 1))(x)
    # Output => (40, 25, 1)

    # Deconvolution layer
    x = Conv2DTranspose(DEPTH, (5,1), padding="same")(x)
    x = Dropout(DROPOUT)(x)
    x = Activation("relu")(x)
    # Output => (20, 25, 1)

    # Unpooling layer
    x = UpSampling2D(size=((2, 1)))(x)
    # Output => (20, 50, 1)

    x = Conv2DTranspose(int(DEPTH/4), (5,1), padding="same")(x)
    x = Dropout(DROPOUT)(x)
    degrees = Activation("relu")(x)
    # Output -> (5,50,1)
    # Construct the model with the input and the output
    G = Model(base_pairs, degrees)

    # Compile the constyructed model
    G.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # 1. optimizer parameter is to choose the stochastic gradient descent algorithm.
    # 2. loss parameter is to choose the loss function.
    # 3. The metrics parameter is to choose the performance metric.

    return G
# Constructs the generator
my_generator = build_my_generator()
my_generator.summary()
## From (BATCH_SIZE, 4, 50, 1) to (BATCH_SIZE, 5, 50, 1)

# Create sequences (#batch_size) from random nucleotides in order to probe the generative network
def noise_nucleotides(batch_size):
    global NUM_NUCLEOTIDES,TYPE_NUCLEOTIDES
    noise = np.zeros([batch_size, TYPE_NUCLEOTIDES, NUM_NUCLEOTIDES])
    for i in range(0, batch_size):
        # Create random nucleotides (A,G,C,T) = (1,2,3,4)
        value = np.random.choice((1, 2, 3, 4))
        j=0
        while j<(NUM_NUCLEOTIDES):
            if value==1:
                noise[i,0,j]=1
            elif value==2:
                noise[i,1,j]=1
            elif value==3:
                noise[i,2,j]=1
            elif value==4:
                noise[i,3,j]=1
            j=j+1
    return noise.reshape((batch_size,TYPE_NUCLEOTIDES,NUM_NUCLEOTIDES,1)) #(BATCH_SIZE,4,50,1)

noise=noise_nucleotides(BATCH_SIZE)
# print(noise)
# print(noise.shape) 
# (BATCH_SIZE, 4, 50, 1)
first_generated_data = my_generator.predict(noise)
print("First generated data: %s" %first_generated_data)
# print(first_generated_data.shape)
# (BATCH_SIZE, 5, 50, 1)

### DISCRIMINATOR
# Creates the discriminator model
# It has 2 inputs and one output
def build_my_discriminator(data_col=NUM_COL_DIS, data_col_gen=NUM_COL_GEN,data_row=NUM_NUCLEOTIDES, channels=NUM_CHAN_DIS, channels_gen=NUM_CHAN_GEN, batch_size=BATCH_SIZE, angles=NUM_ANGLES):

    global DEPTH, DROPOUT

    # It has 2 inputs -> 2 networks that are joint after
    
    # First network: Degrees network 
    # CNN Network

    # Define two sets of inputs 
    # img = Input(shape=(channels, data_row, data_col))
    input_A_sequence = Input(shape=(channels_gen,data_row,data_col_gen))
    input_B_degrees = Input(shape=(channels, data_row, data_col))
    
    # Convolution layer -> DEGREES
    # Input -> (5,50,1)
    x = Conv2D(DEPTH, (5,1), data_format="channels_first", padding="same")(input_B_degrees)
    x = Dropout(DROPOUT)(x)
    x = Activation("relu")(x)
    # Output -> (20,50,1)

    x = MaxPooling2D(pool_size=(2, 1))(x)
    # Output -> (20,25,1)

    x = Conv2D(DEPTH*2, (5,1), padding="same")(x)
    x = Dropout(DROPOUT)(x)
    x = Activation("relu")(x)
    # Output -> (40,25,1)

    x = MaxPooling2D(pool_size=(5, 1))(x)
    # Output -> (40,5,1)

    x = Conv2D(DEPTH*4, (5,1), padding="same")(x)
    x = Dropout(DROPOUT)(x)
    x = Activation("relu")(x)
    # Output -> (80,5,1)

    x = MaxPooling2D(pool_size=(5, 1))(x)
    # Output -> (80,1,1)
    
    # Flatten network
    # It pools the images into a continuous vector through Flattening. 
    # It takes the 2-D array, and converts them to a one dimensional single vector.
    x = Flatten()(x)
    # Output -> 80

    # Fully connected layer: Dense
    # It adds a fully connected layer, 
    # units define the number of nodes that should be present in this layer
    x = Dense(units = int(DEPTH*2))(x)
    x = Activation("relu")(x)
    x = Dropout(DROPOUT)(x)
    # Output -> 40

    x = Dense(units = DEPTH)(x)
    x = Activation("relu")(x)
    degrees_output = Dropout(DROPOUT)(x)
    # Output -> 20

    # Second network: Sequence network => Sequence output
    # CNN network 

    # Convolution layer
    # Input -> (5,50,1)
    y = Conv2D(DEPTH, (5,1), data_format="channels_first", padding="same")(input_A_sequence)
    y = Dropout(DROPOUT)(y)
    y = Activation("relu")(y)
    # Output -> (20,50,1)

    y = MaxPooling2D(pool_size=(2, 1))(y)
    # Output -> (20,25,1)

    y = Conv2D(DEPTH*2, (5,1), padding="same")(y)
    y = Dropout(DROPOUT)(y)
    y = Activation("relu")(y)
    # Output -> (40,25,1)

    y = MaxPooling2D(pool_size=(5, 1))(y)
    # Output -> (40,5,1)

    y = Conv2D(DEPTH*4, (5,1), padding="same")(y)
    y = Dropout(DROPOUT)(y)
    y = Activation("relu")(y)
    # Output -> (80,5,1)

    y = MaxPooling2D(pool_size=(5, 1))(y)
    # Output -> (80,1,1)
    
    # Flatten network
    # It pools the images into a continuous vector through Flattening. 
    # It takes the 2-D array, and converts them to a one dimensional single vector.
    y = Flatten()(y)
    # Output -> 80

    # Fully connected layer: Dense
    # It adds a fully connected layer, 
    # units define the number of nodes that should be present in this layer
    y = Dense(units = DEPTH*2)(y)
    y = Activation("relu")(y)
    y = Dropout(DROPOUT)(y)
    # Output -> 40
    
    y = Dense(units = DEPTH)(y)
    y = Activation("relu")(y)
    sequence_output = Dropout(DROPOUT)(y)
    # Output -> 20

    # Construct both models
    seq = Model(input_A_sequence, sequence_output)
    deg = Model(input_B_degrees, degrees_output)
    
    # Combine the output of the two branches
    combined = concatenate([seq.output, deg.output])
    
    # Apply a Fully-Conected layer and then a regression prediction on the combined outputs
    z = Dense(2, activation="relu")(combined)
    z = Dense(1, activation="relu")(z)
 
    # The model accepts two inputs and the output is single value
    D = Model(inputs=[seq.input, deg.input], outputs=z)

    # Compile the model
    D.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return D

# Constructs the discriminator
my_discriminator = build_my_discriminator()
my_discriminator.summary()

first_prediction = my_discriminator.predict([noise,first_generated_data]) 
# Input: Sequence, degrees = [(2, 1, 50, 2)(2, 5 ,50, 1)]
# Output: (2,1)
# print(first_prediction.shape)
# print(first_prediction)

### GAN
# The discriminator is freeze in GAN model,
# because in the train, it only trains the generative part, the discriminative is trained outside the GAN
def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

# Creates the GAN model
def build_mygan(generator, discriminator, data_col=NUM_COL_GEN, data_row=NUM_NUCLEOTIDES, channels=NUM_CHAN_GEN):
    set_trainability(discriminator, False)
    # The input is teh same that the generative network
    GAN_in = Input(shape=(channels, data_row, data_col))
    # Constructs the generator part
    x = generator(GAN_in)
    # Constructs the discriminator part (which input is the sequence and the degrees from the output of the generative network)
    GAN_out = discriminator([GAN_in,x])
    # Contsruct the model
    GAN = Model(GAN_in, GAN_out)
    # The model es compiled
    GAN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return GAN

#Constructs the GAN model
my_gan = build_mygan(generator=my_generator, discriminator=my_discriminator)
my_gan.summary()
## From (BATCH_SIZE, 4, 50, 2) to (BATCH_SIZE, 1)

#print(noise.shape)
first_gan_data = my_gan.predict(noise)
#print("First GAN data: %s" %first_gan_data)
#print(first_gan_data.shape)

### SAVE PDBs AND CSVs
# Save the given sequence in a csv = 1 line contain: Sequence, Degrees and Score
def save_to_csv(seq, x, score):
    headers = ("Sequence", "Degrees", "Score")
    table = [seq, x, score]
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
def save_created_figures(epoch,my_generative,X_test):
    #noise = noise_nucleotides(1)
    idx = np.random.randint(0, X_test.shape[0], 1)
    X_test=X_test[idx]

    gen_img = my_generative.predict(X_test)
    pose_new = Pose()
    sequence = ""
    global NUM_NUCLEOTIDES
    for no in range(0, NUM_NUCLEOTIDES):
        if X_test[0, 0, no, 0] == 1:
            sequence = sequence+"A[ADE]"
        elif X_test[0, 1, no,0] == 1:
            sequence = sequence+"G[GUA]"
        elif X_test[0, 2, no] == 1:
            sequence = sequence+"C[CYT]"
        elif X_test[0, 3, no] == 1:
            sequence = sequence+"T[THY]"
    pose_new = pose_from_sequence(sequence)

    data=[]
    for k in range(0, NUM_NUCLEOTIDES):
        data.append(gen_img[0,0,k,0])
        pose_new.set_gamma(k+1, gen_img[0,0,k,0])
        data.append(gen_img[0,1,k,0])
        pose_new.set_epsilon(k+1, gen_img[0,1,k,0])
        data.append(gen_img[0,2,k,0])
        pose_new.set_delta(k+1, gen_img[0,2,k,0])
        data.append(gen_img[0,3,k,0])
        pose_new.set_chi(k+1, gen_img[0,3,k,0])
        data.append(gen_img[0,4,k,0])
        pose_new.set_zeta(k+1, gen_img[0,4,k,0])
    score = scorefxn(pose_new)
    save_to_csv(sequence, data, score)
    create_pdb_and_save(pose_new, score, epoch)

### FUNCTION THAT MEASURES THE PERFORMANCE OF THE TRAIN 
# AND SAVES A RESULTS OF THE GENERATIVE NETWORK EACH SAMPLE_INTERVAL EPOCHS
def summarize_performance(epoch,X_test,Y_test,generative,discriminative, batch_size):
    valid = np.zeros((batch_size, 1))
    fake = np.ones((batch_size, 1))
    # Takes some test data and evaluate the discriminative network at this point (# epoch)
    idx = np.random.randint(0, X_test.shape[0], batch_size)
    sequences_DB = X_test[idx]
    degrees_DB = Y_test[idx]
    # For real data
    _, acc_real = discriminative.evaluate(x=[sequences_DB, degrees_DB],y=valid, verbose=0)
    # And wrong data
    degrees_GEN = generative.predict(sequences_DB)
    _, acc_fake = discriminative.evaluate(x=[sequences_DB, degrees_GEN],y=fake, verbose=0) 
    #Verbose=1 -> Progress bar

    # Summarize discriminator performance: 
    print(f"Accuracy real: {acc_real*100}% , fake: {acc_fake*100}%")
    save_created_figures(epoch=epoch,my_generative=generative,X_test=X_test)
    

### TRAIN
# Train the model
def train(X_train,Y_train,Y_test,X_test, generative, discriminative, GAN, epochs=EPOCHS, batch_size=BATCH_SIZE, sample_interval=SAMPLE_INTERVAL):
    # Adversarial ground truths -> in order t cretate the training set for teh discriminator
    valid = np.zeros((batch_size, 1)) #-> For to database images
    fake = np.ones((batch_size, 1)) #-> For to generated images with the generative network

    for epoch in range(epochs):
        # Discriminator
        # Select a random batch of images from the database
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        sequences_DB = X_train[idx]
        degrees_DB = Y_train[idx]

        # Generate a batch of new images from the known sequences with the generator
        degrees_GEN = generative.predict(sequences_DB)

        # Generates training set for the discriminator: takes generated and original images 
        # with its sentesnces and assign the valid or not valid digit.
        # And train the discriminator and update its model weights
        
        my_discriminator.fit(x=[sequences_DB,degrees_DB],y=valid,batch_size=batch_size,epochs=1,verbose=1)
        my_discriminator.fit(x=[sequences_DB,degrees_GEN],y=fake,batch_size=batch_size,epochs=1,verbose=1)
        
        #d_loss_real,_ = my_discriminator.train_on_batch([sequences_DB,degrees_DB], valid)
        #d_loss_fake,_ = my_discriminator.train_on_batch([sequences_DB,degrees_GEN], fake)
        #d_loss = 0.5 *(d_loss_real + d_loss_fake)

        # Generator
        # Train the generator (to have the discriminator label samples as valid)
        # and update its weights
        #g_loss, _ = GAN.train_on_batch(sequences_DB, valid)
        GAN.fit(x=sequences_DB,y=valid,batch_size=batch_size,epochs=1,verbose=0)

        # Plot the progress:
        #print("> Epoch: %d, D loss: %.3f G loss:%.3f" %(epoch, d_loss, g_loss))
    
        # If at save interval => save a generated image sample in this step and plot the accuracy
        if (epoch % sample_interval == 0) or (epoch == (epochs-1)):
            summarize_performance(epoch=epoch,generative=generative, discriminative=discriminative, X_test=X_test, Y_test=Y_test, batch_size=(int(0.1*BATCH_SIZE)+1))

train(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, generative=my_generator, discriminative=my_discriminator, GAN=my_gan)

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

# Evaluate loaded models on test data:
# Fisrt the models are compiled
my_gan_new.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
my_generator_new.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
my_discriminator_new.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Then the models are used with a random nucleotides data => checks that the model is loaded and works 
noise=noise_nucleotides(1)
new_gan = my_gan_new.predict(noise)
print(new_gan)

new_value = my_generator_new.predict(noise)
print(new_value)

new_pred = my_discriminator_new.predict([noise,new_value])
print(new_pred)