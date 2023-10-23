import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

buffer_size = 60
path = os.path.dirname(os.path.abspath(__file__))


def process_video(vid_name):
    TP_avg = 0
    FP_avg = 0
    FN_avg = 0
    TN_avg = 0

    TP_med = 0
    FP_med = 0
    FN_med = 0
    TN_med = 0
    # create a buffer to store the last 60 frames
    buffer = np.zeros((buffer_size, 240, 360), dtype=np.uint8)

    # path to the video file
    vid_path = os.path.join(path, vid_name, "input")

    # path to the ground truth file
    gt_path = os.path.join(path, vid_name, "groundtruth")

    # path to the temporal ROI file
    roi_path = os.path.join(path, vid_name, "temporalROI.txt")

    # read the temporal ROI
    with open(roi_path, "r") as f:
        line = f.readline()
        start_frame, end_frame = line.split()
        roi_start = int(start_frame)
        roi_end = int(end_frame)

    # iterate over the frames in the video
    for frame_idx in range(100, 1100):
        # read the frame
        frame = cv2.imread(os.path.join(vid_path, "in%06d.jpg" % (frame_idx + 1)))

        if frame is None:
            break
        # convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # add the frame to the buffer
        buffer[frame_idx % buffer_size] = frame
        # create background model by averaging the buffer
        background_avg = np.mean(buffer, axis=0).astype(np.uint8)
        # create background model by median filtering the buffer
        background_median = np.median(buffer, axis=0).astype(np.uint8)
        # compute the difference between the current frame and the background model
        diff_avg = cv2.absdiff(frame, background_avg)
        diff_median = cv2.absdiff(frame, background_median)

        # binarize the image
        ret, diff_avg = cv2.threshold(diff_avg, 30, 255, cv2.THRESH_OTSU)
        ret, diff_median = cv2.threshold(diff_median, 30, 255, cv2.THRESH_OTSU)

        # perform a series of erosions and dilations to remove
        # any small blobs of noise from the thresholded image
        diff_avg = cv2.erode(diff_avg, None, iterations=3)
        diff_avg = cv2.dilate(diff_avg, None, iterations=3)

        diff_median = cv2.erode(diff_median, None, iterations=3)
        diff_median = cv2.dilate(diff_median, None, iterations=3)

        # concatenate the images to display them side by side
        display = np.concatenate((frame, diff_avg, diff_median), axis=1)

        if roi_start <= frame_idx <= roi_end:
            # read the ground truth
            ground_truth = cv2.imread(
                f"{gt_path}/gt%06d.png" % (frame_idx + 1), cv2.IMREAD_GRAYSCALE
            )

            # compute the indicators for average background model
            TP_M = np.logical_and((diff_avg == 255), (ground_truth == 255))
            TP_S = np.sum(
                TP_M
            )  # sum of the elements in the matrix TP = TP + TP_S # update of the global indicator
            TP_avg += TP_S
            FP_M = np.logical_and((diff_avg == 255), (ground_truth == 0))
            FP_S = np.sum(
                FP_M
            )  # sum of the elements in the matrix FP = FP + FP_S # update of the global indicator
            FP_avg += FP_S
            FN_M = np.logical_and((diff_avg == 0), (ground_truth == 255))
            FN_S = np.sum(
                FN_M
            )  # sum of the elements in the matrix FN = FN + FN_S # update of the global indicator
            FN_avg += FN_S
            TN_M = np.logical_and((diff_avg == 0), (ground_truth == 0))
            TN_S = np.sum(
                TN_M
            )  # sum of the elements in the matrix TN = TN + TN_S # update of the global indicator
            TN_avg += TN_S

            # compute the indicators for median background model
            TP_M = np.logical_and((diff_median == 255), (ground_truth == 255))
            TP_S = np.sum(
                TP_M
            )  # sum of the elements in the matrix TP = TP + TP_S # update of the global indicator
            TP_med += TP_S
            FP_M = np.logical_and((diff_median == 255), (ground_truth == 0))
            FP_S = np.sum(
                FP_M
            )  # sum of the elements in the matrix FP = FP + FP_S # update of the global indicator
            FP_med += FP_S
            FN_M = np.logical_and((diff_median == 0), (ground_truth == 255))
            FN_S = np.sum(
                FN_M
            )  # sum of the elements in the matrix FN = FN + FN_S # update of the global indicator
            FN_med += FN_S
            TN_M = np.logical_and((diff_median == 0), (ground_truth == 0))
            TN_S = np.sum(
                TN_M
            )  # sum of the elements in the matrix TN = TN + TN_S # update of the global indicator
            TN_med += TN_S

        # display the image
        cv2.imshow("frame", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # print the precision, recall and F1 score for the average background model
    print("Average background model")
    print("Precision: ", TP_avg / (TP_avg + FP_avg))
    print("Recall: ", TP_avg / (TP_avg + FN_avg))
    print("F1 score: ", 2 * TP_avg / (2 * TP_avg + FP_avg + FN_avg))

    # print the precision, recall and F1 score for the median background model
    print("Median background model")
    print("Precision: ", TP_med / (TP_med + FP_med))
    print("Recall: ", TP_med / (TP_med + FN_med))
    print("F1 score: ", 2 * TP_med / (2 * TP_med + FP_med + FN_med))


def process_video_approx(vid_name, alpha=0.01):
    TP_avg = 0
    FP_avg = 0
    FN_avg = 0
    TN_avg = 0

    TP_med = 0
    FP_med = 0
    FN_med = 0
    TN_med = 0

    # path to the video file
    vid_path = os.path.join(path, vid_name, "input")

    # path to the ground truth file
    gt_path = os.path.join(path, vid_name, "groundtruth")

    # path to the temporal ROI file
    roi_path = os.path.join(path, vid_name, "temporalROI.txt")

    # read the temporal ROI
    with open(roi_path, "r") as f:
        line = f.readline()
        start_frame, end_frame = line.split()
        roi_start = int(start_frame)
        roi_end = int(end_frame)

    # create the binary background model using first frame
    frame = cv2.imread(f"{vid_path}/in%06d.jpg" % 1, cv2.IMREAD_GRAYSCALE)
    background_avg = frame.astype(np.float32)
    background_median = frame.astype(np.float32)

    for frame_idx in range(roi_start, roi_end):
        frame = cv2.imread(
            f"{vid_path}/in%06d.jpg" % (frame_idx + 1), cv2.IMREAD_GRAYSCALE
        )
        # approximate the average background model using the current frame and the previous average background model
        # with a weight alpha and current frame
        background_avg = alpha * frame + (1 - alpha) * background_avg

        # approximate the median background model using the current frame and the previous median background model
        # with a weight alpha and current frame
        background_median = np.where(
            background_median > frame,
            background_median - 1,
            np.where(
                background_median < frame, background_median + 1, background_median
            ),
        )

        # compute the difference between the current frame and the background model
        diff_avg = cv2.absdiff(frame, background_avg.astype(np.uint8))
        diff_median = cv2.absdiff(frame, background_median.astype(np.uint8))

        # threshold the difference image
        _, diff_avg = cv2.threshold(diff_avg, 30, 255, cv2.THRESH_BINARY)
        _, diff_median = cv2.threshold(diff_median, 30, 255, cv2.THRESH_BINARY)

        # perform erosion and dilation to remove noise
        diff_avg = cv2.erode(diff_avg, None, iterations=2)
        diff_avg = cv2.dilate(diff_avg, None, iterations=4)

        diff_median = cv2.erode(diff_median, None, iterations=2)
        diff_median = cv2.dilate(diff_median, None, iterations=4)

        if roi_start <= frame_idx <= roi_end:
            # read the ground truth
            ground_truth = cv2.imread(
                f"{gt_path}/gt%06d.png" % (frame_idx + 1), cv2.IMREAD_GRAYSCALE
            )

            # compute the indicators for average background model
            TP_M = np.logical_and((diff_avg == 255), (ground_truth == 255))
            TP_S = np.sum(
                TP_M
            )  # sum of the elements in the matrix TP = TP + TP_S # update of the global indicator
            TP_avg += TP_S
            FP_M = np.logical_and((diff_avg == 255), (ground_truth == 0))
            FP_S = np.sum(
                FP_M
            )  # sum of the elements in the matrix FP = FP + FP_S # update of the global indicator
            FP_avg += FP_S
            FN_M = np.logical_and((diff_avg == 0), (ground_truth == 255))
            FN_S = np.sum(
                FN_M
            )  # sum of the elements in the matrix FN = FN + FN_S # update of the global indicator
            FN_avg += FN_S
            TN_M = np.logical_and((diff_avg == 0), (ground_truth == 0))
            TN_S = np.sum(
                TN_M
            )  # sum of the elements in the matrix TN = TN + TN_S # update of the global indicator
            TN_avg += TN_S

            # compute the indicators for median background model
            TP_M = np.logical_and((diff_median == 255), (ground_truth == 255))
            TP_S = np.sum(
                TP_M
            )  # sum of the elements in the matrix TP = TP + TP_S # update of the global indicator
            TP_med += TP_S
            FP_M = np.logical_and((diff_median == 255), (ground_truth == 0))
            FP_S = np.sum(
                FP_M
            )  # sum of the elements in the matrix FP = FP + FP_S # update of the global indicator
            FP_med += FP_S
            FN_M = np.logical_and((diff_median == 0), (ground_truth == 255))
            FN_S = np.sum(
                FN_M
            )  # sum of the elements in the matrix FN = FN + FN_S # update of the global indicator
            FN_med += FN_S
            TN_M = np.logical_and((diff_median == 0), (ground_truth == 0))
            TN_S = np.sum(
                TN_M
            )  # sum of the elements in the matrix TN = TN + TN_S # update of the global indicator
            TN_med += TN_S

        # concatenate the current frame and the difference image
        display = np.concatenate((frame, diff_avg, diff_median), axis=1)

        # display the image
        cv2.imshow("frame", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # print the precision, recall and F1 score for the average background model
    print("Average background model")
    print("Precision: ", TP_avg / (TP_avg + FP_avg))
    print("Recall: ", TP_avg / (TP_avg + FN_avg))
    print("F1 score: ", 2 * TP_avg / (2 * TP_avg + FP_avg + FN_avg))

    # print the precision, recall and F1 score for the median background model
    print("Median background model")
    print("Precision: ", TP_med / (TP_med + FP_med))
    print("Recall: ", TP_med / (TP_med + FN_med))
    print("F1 score: ", 2 * TP_med / (2 * TP_med + FP_med + FN_med))


def process_video_conservative(vid_name, alpha=0.01):
    TP_avg = 0
    FP_avg = 0
    FN_avg = 0
    TN_avg = 0

    TP_med = 0
    FP_med = 0
    FN_med = 0
    TN_med = 0

    # path to the video file
    vid_path = os.path.join(path, vid_name, "input")

    # path to the ground truth file
    gt_path = os.path.join(path, vid_name, "groundtruth")

    # path to the temporal ROI file
    roi_path = os.path.join(path, vid_name, "temporalROI.txt")

    # read the temporal ROI
    with open(roi_path, "r") as f:
        line = f.readline()
        start_frame, end_frame = line.split()
        roi_start = int(start_frame)
        roi_end = int(end_frame)

    # create the binary background model using first frame
    frame = cv2.imread(f"{vid_path}/in%06d.jpg" % 1, cv2.IMREAD_GRAYSCALE)
    background_avg = frame.astype(np.float32)
    background_median = frame.astype(np.float32)

    for frame_idx in range(roi_start, roi_end):
        frame = cv2.imread(
            f"{vid_path}/in%06d.jpg" % (frame_idx + 1), cv2.IMREAD_GRAYSCALE
        )
        # approximate the average background model using the current frame and the previous average background model
        # with a weight alpha and current frame, also update only the pixels that are classified as background
        background_avg = np.where(
            cv2.absdiff(background_avg.astype(np.uint8), frame) < 5,
            alpha * frame + (1 - alpha) * background_avg,
            background_avg,
        )

        # approximate the median background model using the current frame and the previous median background model
        # with a weight alpha and current frame, also update only the pixels that are classified as background
        background_median = np.where(
            cv2.absdiff(background_median.astype(np.uint8), frame) < 5,
            np.where(
                background_median > frame,
                background_median - 1,
                np.where(
                    background_median < frame, background_median + 1, background_median
                ),
            ),
            background_median,
        )

        # compute the difference between the current frame and the background model
        diff_avg = cv2.absdiff(frame, background_avg.astype(np.uint8))
        diff_median = cv2.absdiff(frame, background_median.astype(np.uint8))

        # threshold the difference image
        _, diff_avg = cv2.threshold(diff_avg, 30, 255, cv2.THRESH_BINARY)
        _, diff_median = cv2.threshold(diff_median, 30, 255, cv2.THRESH_BINARY)

        # perform erosion and dilation to remove noise
        diff_avg = cv2.erode(diff_avg, None, iterations=2)
        diff_avg = cv2.dilate(diff_avg, None, iterations=4)

        diff_median = cv2.erode(diff_median, None, iterations=2)
        diff_median = cv2.dilate(diff_median, None, iterations=4)

        if roi_start <= frame_idx <= roi_end:
            # read the ground truth
            ground_truth = cv2.imread(
                f"{gt_path}/gt%06d.png" % (frame_idx + 1), cv2.IMREAD_GRAYSCALE
            )

            # compute the indicators for average background model
            TP_M = np.logical_and((diff_avg == 255), (ground_truth == 255))
            TP_S = np.sum(
                TP_M
            )  # sum of the elements in the matrix TP = TP + TP_S # update of the global indicator
            TP_avg += TP_S
            FP_M = np.logical_and((diff_avg == 255), (ground_truth == 0))
            FP_S = np.sum(
                FP_M
            )  # sum of the elements in the matrix FP = FP + FP_S # update of the global indicator
            FP_avg += FP_S
            FN_M = np.logical_and((diff_avg == 0), (ground_truth == 255))
            FN_S = np.sum(
                FN_M
            )  # sum of the elements in the matrix FN = FN + FN_S # update of the global indicator
            FN_avg += FN_S
            TN_M = np.logical_and((diff_avg == 0), (ground_truth == 0))
            TN_S = np.sum(
                TN_M
            )  # sum of the elements in the matrix TN = TN + TN_S # update of the global indicator
            TN_avg += TN_S

            # compute the indicators for median background model
            TP_M = np.logical_and((diff_median == 255), (ground_truth == 255))
            TP_S = np.sum(
                TP_M
            )  # sum of the elements in the matrix TP = TP + TP_S # update of the global indicator
            TP_med += TP_S
            FP_M = np.logical_and((diff_median == 255), (ground_truth == 0))
            FP_S = np.sum(
                FP_M
            )  # sum of the elements in the matrix FP = FP + FP_S # update of the global indicator
            FP_med += FP_S
            FN_M = np.logical_and((diff_median == 0), (ground_truth == 255))
            FN_S = np.sum(
                FN_M
            )  # sum of the elements in the matrix FN = FN + FN_S # update of the global indicator
            FN_med += FN_S
            TN_M = np.logical_and((diff_median == 0), (ground_truth == 0))
            TN_S = np.sum(
                TN_M
            )  # sum of the elements in the matrix TN = TN + TN_S # update of the global indicator
            TN_med += TN_S

        # concatenate the current frame and the difference image
        display = np.concatenate((frame, diff_avg, diff_median), axis=1)

        # display the image
        cv2.imshow("frame", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # print the precision, recall and F1 score for the average background model
    print("Average background model")
    print("Precision: ", TP_avg / (TP_avg + FP_avg))
    print("Recall: ", TP_avg / (TP_avg + FN_avg))
    print("F1 score: ", 2 * TP_avg / (2 * TP_avg + FP_avg + FN_avg))

    # print the precision, recall and F1 score for the median background model
    print("Median background model")
    print("Precision: ", TP_med / (TP_med + FP_med))
    print("Recall: ", TP_med / (TP_med + FN_med))
    print("F1 score: ", 2 * TP_med / (2 * TP_med + FP_med + FN_med))

def process_video_gmm(vid_name, alpha=0.001):
    """Process a video using the GMM background model.

    Args:
        vid_name (str): name of the video to process
        alpha (float): learning rate for the GMM background model
    """
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    # create the background model
    background_model = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=False
    )

    # path to the video file
    vid_path = os.path.join(path, vid_name, "input")

    # path to the ground truth file
    gt_path = os.path.join(path, vid_name, "groundtruth")

    # path to the temporal ROI file
    roi_path = os.path.join(path, vid_name, "temporalROI.txt")

    # read the temporal ROI
    with open(roi_path, "r") as f:
        line = f.readline()
        start_frame, end_frame = line.split()
        roi_start = int(start_frame)
        roi_end = int(end_frame)

    # display the video with the foreground mask
    for frame_idx in range(roi_start, roi_end):
        # read the next frame
        frame = cv2.imread(
            f"{vid_path}/in%06d.jpg" % (frame_idx + 1), cv2.IMREAD_GRAYSCALE
        )

        # update the background model
        background_model.apply(frame, learningRate=alpha)

        # using createBackgroundSubtractorMOG2, the background model is stored in the
        # backgroundModel attribute
        background = background_model.getBackgroundImage()

        # compute the difference between the current frame and the background model
        diff = cv2.absdiff(frame, background)

        # threshold the difference image
        _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # perform erosion and dilation to remove noise
        diff = cv2.erode(diff, None, iterations=2)
        diff = cv2.dilate(diff, None, iterations=4)

        # read the ground truth
        ground_truth = cv2.imread(
            f"{gt_path}/gt%06d.png" % (frame_idx + 1), cv2.IMREAD_GRAYSCALE
        )

        # compute the indicators
        TP_M = np.logical_and((diff == 255), (ground_truth == 255))
        TP_S = np.sum(
            TP_M
        )
        TP += TP_S
        FP_M = np.logical_and((diff == 255), (ground_truth == 0))
        FP_S = np.sum(
            FP_M
        )
        FP += FP_S
        FN_M = np.logical_and((diff == 0), (ground_truth == 255))
        FN_S = np.sum(
            FN_M
        )
        FN += FN_S
        TN_M = np.logical_and((diff == 0), (ground_truth == 0))
        TN_S = np.sum(
            TN_M
        )
        TN += TN_S


        # concatenate the current frame and the difference image
        # diff need to be converted to color to be concatenated with the frame
        # diff_display = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        display = np.concatenate((frame, diff), axis=1)

        # display the image
        cv2.imshow("frame", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # print the precision, recall and F1 score
    print("GMM background model")
    print("Precision: ", TP / (TP + FP))
    print("Recall: ", TP / (TP + FN))
    print("F1 score: ", 2 * TP / (2 * TP + FP + FN))


def process_video_knn(vid_name, alpha=0.001):
    """Process a video using the KNN background model.

    Args:
        vid_name (str): name of the video to process
        alpha (float): learning rate for the KNN background model
    """
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    # create the background model
    background_model = cv2.createBackgroundSubtractorKNN(
        history=20, dist2Threshold=400.0, detectShadows=False
    )

    # path to the video file
    vid_path = os.path.join(path, vid_name, "input")

    # path to the ground truth file
    gt_path = os.path.join(path, vid_name, "groundtruth")

    # path to the temporal ROI file
    roi_path = os.path.join(path, vid_name, "temporalROI.txt")

    # read the temporal ROI
    with open(roi_path, "r") as f:
        line = f.readline()
        start_frame, end_frame = line.split()
        roi_start = int(start_frame)
        roi_end = int(end_frame)

    # display the video with the foreground mask
    for frame_idx in range(roi_start, roi_end):
        # read the next frame
        frame = cv2.imread(
            f"{vid_path}/in%06d.jpg" % (frame_idx + 1), cv2.IMREAD_GRAYSCALE
        )

        # update the background model
        background_model.apply(frame, learningRate=alpha)

        # using createBackgroundSubtractorKNN, the background model is stored in the
        # backgroundModel attribute
        background = background_model.getBackgroundImage()

        # compute the difference between the current frame and the background model
        diff = cv2.absdiff(frame, background)

        # threshold the difference image
        _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # perform erosion and dilation to remove noise
        diff = cv2.erode(diff, None, iterations=2)
        diff = cv2.dilate(diff, None, iterations=4)

        # read the ground truth
        ground_truth = cv2.imread(
            f"{gt_path}/gt%06d.png" % (frame_idx + 1), cv2.IMREAD_GRAYSCALE
        )

        # compute the indicators
        TP_M = np.logical_and((diff == 255), (ground_truth == 255))
        TP_S = np.sum(
            TP_M
        )
        TP += TP_S
        FP_M = np.logical_and((diff == 255), (ground_truth == 0))
        FP_S = np.sum(
            FP_M
        )
        FP += FP_S
        FN_M = np.logical_and((diff == 0), (ground_truth == 255))
        FN_S = np.sum(
            FN_M
        )
        FN += FN_S
        TN_M = np.logical_and((diff == 0), (ground_truth == 0))
        TN_S = np.sum(
            TN_M
        )
        TN += TN_S

        # concatenate the current frame and the difference image
        # diff need to be converted to color to be concatenated with the frame
        # diff_display = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        display = np.concatenate((frame, diff), axis=1)

        # display the image
        cv2.imshow("frame", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # print the precision, recall and F1 score
    print("KNN background model")
    print("Precision: ", TP / (TP + FP))
    print("Recall: ", TP / (TP + FN))
    print("F1 score: ", 2 * TP / (2 * TP + FP + FN))


if __name__ == "__main__":
    # process_video("pedestrian")
    # process_video_approx("office", alpha=0.01)
    # process_video_conservative("office", alpha=0.01)
    # process_video_gmm("office", alpha=0.003)
    process_video_knn("office", alpha=0.01)