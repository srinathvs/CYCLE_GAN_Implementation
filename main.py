# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
from glob import glob
import cv2
import joblib
import numpy as np
import tensorflow
import random
from numpy import add
from matplotlib import pyplot as plt
from skimage.io import imread
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.optimizer_v2.adam import Adam

import RTX_GAN
from RTX_GAN import Discriminator, Generator


def load_images(data_dir):
    imagesA = glob(data_dir + '\\Normal\\*.*')
    imagesB = glob(data_dir + '/RTX/*.*')

    allImagesA = []
    allImagesB = []

    for index, filename in enumerate(imagesA):
        if index < 100:
            imgA = imread(filename, pilmode='RGB')
            imgB = imread(imagesB[index], pilmode='RGB')
            print(filename)
            if np.random.random() > 0.5:
                imgA = np.fliplr(imgA)
                imgB = np.fliplr(imgB)

            allImagesA.append(imgA)
            allImagesB.append(imgB)

    # Normalize images
    allImagesA = np.array(allImagesA) / 127.5 - 1.
    allImagesB = np.array(allImagesB) / 127.5 - 1.

    return allImagesA, allImagesB


def load_continued_images(data_dir):
    imagesA = glob(data_dir + '\\Normal\\*.*')
    imagesB = glob(data_dir + '/RTX/*.*')

    allImagesA = []
    allImagesB = []

    randomlist = []
    for i in range(0, 50):
        n = random.randint(1, len(imagesA))
        randomlist.append(n)

    for index, filename in enumerate(imagesA):
        if index in randomlist:
            imgA = imread(filename, pilmode='RGB')
            imgB = imread(imagesB[index], pilmode='RGB')
            print(filename)
            if np.random.random() > 0.5:
                imgA = np.fliplr(imgA)
                imgB = np.fliplr(imgB)

            allImagesA.append(imgA)
            allImagesB.append(imgB)

    # Normalize images
    allImagesA = np.array(allImagesA) / 127.5 - 1.
    allImagesB = np.array(allImagesB) / 127.5 - 1.

    return allImagesA, allImagesB


def train_model():
    batch_size = 1
    epochs = 400
    # create_buffers()

    allImgA, allImgB = load_images('DataFrames')

    common_optimizer = Adam(.0002, .5)

    # building discriminators
    base_DiscriminatorA = Discriminator(height=960, width=540)
    base_DiscriminatorB = Discriminator(height=960, width=540)

    base_Generator_A_to_B = Generator(height=960, width=540)
    base_Generator_B_to_A = Generator(height=960, width=540)

    discriminatorA = base_DiscriminatorA.build_model()
    discriminatorB = base_DiscriminatorB.build_model()

    discriminatorA.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])
    discriminatorB.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

    discriminatorA.summary()

    generator_AtoB = base_Generator_A_to_B.build_model_generator()
    generator_BtoA = base_Generator_B_to_A.build_model_generator()

    input_A = Input(shape=(540, 960, 3))
    input_B = Input(shape=(540, 960, 3))

    generatedB = generator_AtoB(input_A)
    generatedA = generator_BtoA(input_B)

    generator_AtoB.summary()

    reconA = generator_BtoA(generatedB)
    reconB = generator_AtoB(generatedA)

    generatedAID = generator_BtoA(input_A)
    generatedBID = generator_AtoB(input_B)

    discriminatorA.trainable = False
    discriminatorB.trainable = False

    probs_A = discriminatorA(generatedA)
    probs_B = discriminatorB(generatedB)

    RTX_GAN_model = Model(inputs=[input_A, input_B],
                          outputs=[probs_A, probs_B, reconA, reconB, generatedAID, generatedBID])

    RTX_GAN_model.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                          loss_weights=[1.0, 1.0, 10.0, 10.0, 1.0, 1.0],
                          optimizer=common_optimizer)

    RTX_GAN_model.summary()
    for epoch in range(1, epochs):
        print("Epoch : {}".format(epoch))

        dis_losses = []
        gen_losses = []

        real_labels = np.ones((batch_size, 32, 59, 1))
        fake_labels = np.zeros((batch_size, 32, 59, 1))

        num_batches = int(min(allImgA.shape[0], allImgB.shape[0]) / batch_size)
        print("Number of batches : {}".format(num_batches))

        for index in range(num_batches):
            print("Batch : {}".format(index))
            batchA = allImgA[index * batch_size: (index + 1) * batch_size]
            batchB = allImgB[index * batch_size: (index + 1) * batch_size]

            generatedB = generator_AtoB.predict(batchA)
            generatedA = generator_BtoA.predict(batchB)

            dALoss1 = discriminatorA.train_on_batch(batchA, real_labels)
            dALoss2 = discriminatorA.train_on_batch(generatedA, fake_labels)

            dBLoss1 = discriminatorB.train_on_batch(batchB, real_labels)
            dBLoss2 = discriminatorB.train_on_batch(generatedB, fake_labels)

            d_loss = .5 * np.add((.5 * np.add(dALoss1, dALoss2)), (.5 * np.add(dBLoss1, dBLoss2)))
            # d_loss = .5 * np.add(dALoss1, dBLoss1)
            g_loss = RTX_GAN_model.train_on_batch([batchA, batchB],
                                                  [real_labels, real_labels, batchA, batchB, batchA, batchB])
            dis_losses.append(d_loss)
            gen_losses.append(g_loss)
            print("d_loss : ", d_loss, " g_loss : ", g_loss)
        # if epoch % 100 == 0:
        #     imagesA, imagesB = load_test_batch(data_dir='DataFrames', batch_size=2)
        #
        #     for index in range(len(imagesA)):
        #         batchA = imagesA[index]
        #         batchB = imagesB[index]
        #         # Generate images
        #         generatedB = generator_AtoB.predict(batchA)
        #         generatedA = generator_BtoA.predict(batchB)
        #
        #         # Get reconstructed images
        #         reconsA = generator_BtoA.predict(generatedB)
        #         reconsB = generator_AtoB.predict(generatedA)
        #
        #         # Save original, generated and reconstructed image
        #         # save_images(originalA=batchA, generatedB=generatedB, recosntructedA=reconsA,
        #         #             originalB=batchB, generatedA=generatedA, reconstructedB=reconsB,
        #         #             path="results/gen_{}_{}".format(epoch, index))
    generator_AtoB.save_weights("generatorAToB.h5")
    generator_BtoA.save_weights("generatorBToA.h5")
    discriminatorA.save_weights("discriminatorA.h5")
    discriminatorB.save_weights("discriminatorB.h5")


def continue_training():
    epochs = 100
    common_optimizer = Adam(.0002, .5)

    allImgA, allImgB = load_continued_images('DataFrames')
    # building discriminators
    base_DiscriminatorA = Discriminator(height=960, width=540)
    base_DiscriminatorB = Discriminator(height=960, width=540)

    base_Generator_A_to_B = Generator(height=960, width=540)
    base_Generator_B_to_A = Generator(height=960, width=540)

    discriminatorA = base_DiscriminatorA.build_model()
    discriminatorB = base_DiscriminatorB.build_model()

    discriminatorA.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])
    discriminatorB.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

    # discriminatorA.summary()

    generator_AtoB = base_Generator_A_to_B.build_model_generator()
    generator_BtoA = base_Generator_B_to_A.build_model_generator()

    generator_AtoB.load_weights('generatorAToB_cont.h5')
    generator_BtoA.load_weights('generatorBToA_cont.h5')
    discriminatorA.load_weights('discriminatorA_cont.h5')
    discriminatorB.load_weights('discriminatorB_cont.h5')

    input_A = Input(shape=(540, 960, 3))
    input_B = Input(shape=(540, 960, 3))

    generatedB = generator_AtoB(input_A)
    generatedA = generator_BtoA(input_B)

    # generator_AtoB.summary()

    reconA = generator_BtoA(generatedB)
    reconB = generator_AtoB(generatedA)

    generatedAID = generator_BtoA(input_A)
    generatedBID = generator_AtoB(input_B)

    discriminatorA.trainable = False
    discriminatorB.trainable = False

    probs_A = discriminatorA(generatedA)
    probs_B = discriminatorB(generatedB)

    RTX_GAN_model = Model(inputs=[input_A, input_B],
                          outputs=[probs_A, probs_B, reconA, reconB, generatedAID, generatedBID])

    RTX_GAN_model.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                          loss_weights=[1.0, 1.0, 10.0, 10.0, 1.0, 1.0],
                          optimizer=common_optimizer)

    # RTX_GAN_model.summary()
    batch_size = 1
    for epoch in range(1, epochs):
        print("Epoch : {}".format(epoch))

        dis_losses = []
        gen_losses = []

        real_labels = np.ones((batch_size, 32, 59, 1))
        fake_labels = np.zeros((batch_size, 32, 59, 1))

        num_batches = int(min(allImgA.shape[0], allImgB.shape[0]) / batch_size)
        print("Number of batches : {}".format(num_batches))

        for index in range(num_batches):
            print("Batch : {}".format(index))
            batchA = allImgA[index * batch_size: (index + 1) * batch_size]
            batchB = allImgB[index * batch_size: (index + 1) * batch_size]

            generatedB = generator_AtoB.predict(batchA)
            generatedA = generator_BtoA.predict(batchB)

            dALoss1 = discriminatorA.train_on_batch(batchA, real_labels)
            dALoss2 = discriminatorA.train_on_batch(generatedA, fake_labels)

            dBLoss1 = discriminatorB.train_on_batch(batchB, real_labels)
            dBLoss2 = discriminatorB.train_on_batch(generatedB, fake_labels)

            d_loss = .5 * np.add((.5 * np.add(dALoss1, dALoss2)), (.5 * np.add(dBLoss1, dBLoss2)))
            # d_loss = .5 * np.add(dALoss1, dBLoss1)
            g_loss = RTX_GAN_model.train_on_batch([batchA, batchB],
                                                  [real_labels, real_labels, batchA, batchB, batchA, batchB])
            dis_losses.append(d_loss)
            gen_losses.append(g_loss)
            print("d_loss : ", d_loss, " g_loss : ", g_loss)
        # if epoch % 100 == 0:
        #     imagesA, imagesB = load_test_batch(data_dir='DataFrames', batch_size=2)
        #
        #     for index in range(len(imagesA)):
        #         batchA = imagesA[index]
        #         batchB = imagesB[index]
        #         # Generate images
        #         generatedB = generator_AtoB.predict(batchA)
        #         generatedA = generator_BtoA.predict(batchB)
        #
        #         # Get reconstructed images
        #         reconsA = generator_BtoA.predict(generatedB)
        #         reconsB = generator_AtoB.predict(generatedA)
        #
        #         # Save original, generated and reconstructed image
        #         # save_images(originalA=batchA, generatedB=generatedB, recosntructedA=reconsA,
        #         #             originalB=batchB, generatedA=generatedA, reconstructedB=reconsB,
        #         #             path="results/gen_{}_{}".format(epoch, index))
    generator_AtoB.save_weights("generatorAToB_cont.h5")
    generator_BtoA.save_weights("generatorBToA_cont.h5")
    discriminatorA.save_weights("discriminatorA_cont.h5")
    discriminatorB.save_weights("discriminatorB_cont.h5")


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    # train_model()
    # genAB = RTX_GAN.Generator(height=960, width=540)
    # disA = RTX_GAN.Discriminator(height=960, width=540)
    # genA_B = genAB.build_model_generator()
    # discrim_A = disA.build_model()
    # genA_B.summary()
    # discrim_A.summary()
    # continue_training()
    check_model()


def check_model():
    path_to_img = "C:\\Users\\srina\\PycharmProjects\\Final_Project_GAN\\DataFrames\\Normal\\witcher27340.png"
    path_to_img2 = "C:\\Users\\srina\\Videos\\Captures\\Age of Empires II_ Definitive Edition 6_5_2021 7_24_44 PM.png"
    path_to_img3 = "C:\\Users\\srina\\Pictures\\Camera Roll\\WIN_20210606_20_40_44_Pro.jpg"
    og_img = cv2.imread(path_to_img3)
    og_img = cv2.resize(og_img, (960, 540))
    cv2.imshow('original', og_img)
    cv2.waitKey(0)
    genAB = RTX_GAN.Generator(height=960, width=540)
    genA_B = genAB.build_model_generator()
    genB_A = genAB.build_model_generator()
    og_img = tensorflow.expand_dims(og_img, axis=0)
    genA_B.load_weights('generatorAToB.h5')
    genB_A.load_weights('generatorBToA.h5')
    RTX_img = genA_B.predict(og_img)
    arr_ = np.squeeze(RTX_img, axis=0)
    og_img2 = genB_A.predict(RTX_img)
    og_img2 = np.squeeze(og_img2, axis=0)
    cv2.imshow('RTX transformed', arr_)
    cv2.waitKey(0)
    cv2.imshow('Recons', og_img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_buffers():
    normal_img()
    rtx_img()


def normal_img():
    pathA = "DataFrames/Normal/"

    imagesA = []

    imgA = glob(pathA)

    for filename in os.listdir(pathA):
        print(filename)
        img_ = imread(os.path.join(pathA, filename))
        imagesA.append(img_)

    joblib.dump(imagesA, "Img_Buffers/imgA.pkl")


def rtx_img():
    pathB = "DataFrames/RTX/"

    imagesB = []

    for filename in os.listdir(pathB):
        print(filename)
        img_ = imread(os.path.join(pathB, filename))
        imagesB.append(img_)

    joblib.dump(imagesB, "Img_Buffers/imgB.pkl")


def load_test_batch(data_dir, batch_size):
    imagesA = glob(data_dir + '/Normal/*.*')
    imagesB = glob(data_dir + '/RTX/*.*')

    imagesA = np.random.choice(imagesA, batch_size)
    imagesB = np.random.choice(imagesB, batch_size)

    allA = []
    allB = []

    for i in range(5000, len(imagesA)):
        if i >= 5000:
            # Load images and resize images
            imgA = imread(imagesA[i], pilmode='RGB').astype(np.float32)
            imgB = imread(imagesB[i], pilmode='RGB').astype(np.float32)

            allA.append(imgA)
            allB.append(imgB)

    return np.array(allA) / 127.5 - 1.0, np.array(allB) / 127.5 - 1.0


def predict_img(frame, generator_obg):
    predicted_frame = generator_obg.predict(frame)
    plt.subplot(111), plt.imshow(predicted_frame), plt.title('predicted')
    plt.show()
    return predicted_frame


def save_images(originalA, generatedB, recosntructedA, originalB, generatedA, reconstructedB, path):
    fig = plt.figure()
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(originalA)
    ax.axis("off")
    ax.set_title("Original")

    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(generatedB)
    ax.axis("off")
    ax.set_title("Generated")

    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(recosntructedA)
    ax.axis("off")
    ax.set_title("Reconstructed")

    ax = fig.add_subplot(2, 3, 4)
    ax.imshow(originalB)
    ax.axis("off")
    ax.set_title("Original")

    ax = fig.add_subplot(2, 3, 5)
    ax.imshow(generatedA)
    ax.axis("off")
    ax.set_title("Generated")

    ax = fig.add_subplot(2, 3, 6)
    ax.imshow(reconstructedB)
    ax.axis("off")
    ax.set_title("Reconstructed")

    plt.savefig(path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    train_model()
    # fake_model = RTX_GAN.fake_model()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
