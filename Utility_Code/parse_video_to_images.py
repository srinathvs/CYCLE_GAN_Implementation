import cv2
import os


def main():
    datapath_class1 = "C:\\Users\\srina\\Downloads\\RTX\\Non-Raytraced"
    datapath_class2 = "C:\\Users\\srina\\Downloads\\RTX\\RayTraced"
    rtx = []
    nonrtx = []

    for filename in os.listdir(datapath_class1):
        file_path = os.path.join(datapath_class1, filename)
        filename = filename.split('.')
        name = filename[0]
        nonrtx.append((file_path, name))
    for filename in os.listdir(datapath_class2):
        file_path = os.path.join(datapath_class2, filename)
        filename = filename.split('.')
        name = filename[0]
        rtx.append((file_path, name))
    #
    # for elem, name in nonrtx:
    #     print(elem, name)
    #     video_read(elem, True, name)

    for elem, name in rtx:
        print(elem, name)
        video_read(elem, False, name)


def video_read(path, flag, filename):
    # extract frames from a video and save to directory as 'x.png' where 
    # x is the frame index
    vid_cap = cv2.VideoCapture(path)

    count = 0
    while vid_cap.isOpened():
        success, image = vid_cap.read()
        if success:
            dim = (int(image.shape[1] / 2), int(image.shape[0] / 2))
            image = cv2.resize(image, dim, cv2.INTER_AREA)
        if success and flag:
            if count % 20 == 0:
                out_path1 = "C:\\Users\\srina\\PycharmProjects\\Final_Project_GAN\\DataFrames\\Normal\\"
                out_path1 += filename + str(count) + '.png'
                cv2.imwrite(out_path1, image)

        elif success and not flag:
            if count % 20 == 0:
                out_path2 = "C:\\Users\\srina\\PycharmProjects\\Final_Project_GAN\\DataFrames\\RTX\\"
                out_path2 += filename + str(count) + '.png'
                cv2.imwrite(out_path2, image)

        else:
            break
        count += 1
    cv2.destroyAllWindows()
    vid_cap.release()


if __name__ == '__main__':
    main()
