import cv2
import numpy as np
import matplotlib.pyplot as plt

def first_task():
    cap = cv2.VideoCapture("vid1_IR.avi")
    while cap.isOpened():
        ret, frame = cap.read()

        # if frame is not read correctly ret is False
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        G = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        # binarize the image
        ret, G = cv2.threshold(G, 127, 255, cv2.THRESH_OTSU)
        # perform a series of openings and closings to remove
        # any small blobs of noise from the thresholded image
        G = cv2.morphologyEx(G, cv2.MORPH_OPEN, None, iterations=2)
        G = cv2.morphologyEx(G, cv2.MORPH_CLOSE, None, iterations=2)
        # index objects in the image using connected components with stats function from opencv
        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(G, connectivity=8)
        # first label is background
        # loop over the number of unique connected component labels
        index = 0
        rects = []
        for i in range(1, numLabels):
            # extract the connected component statistics for the current label
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            # if the width to height ratio is more than 1.5 we exclude the object
            if w / h > 1.5:
                continue
            # if the area is less than 100 pixels we exclude the object
            # if area < 200:
            #     continue
            # cv2.rectangle(G, (x, y), (x + w, y + h), 255, 2)
            # add index number above the rectangle
            index += 1
            # cv2.putText(G, str(index), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
            # add rectangle to the list
            rects.append((x, y, w, h))
            # draw green rectangle around the object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # merge the rectangles that are close to each other and approximately in the same vertical position
        # show also original image
        cv2.imshow("Original", frame)
        if len(rects) == 1:
            final_rects = rects
        else:
            overlapping_dict= {i: [] for i in range(len(rects))}
            # iterate over all the rectangles and if horizontal lines overlap add them to the dictionary
            for i in range(len(rects)):
                for j in range(len(rects)):
                    if i == j:
                        continue
                    if rects[j][0] <= rects[i][0] <= rects[j][0] + rects[j][2] or rects[i][0] <= rects[j][0] <= rects[i][0] + rects[i][2]:
                        overlapping_dict[i].append(j)

            # create a list of rectangles created by merging the overlapping rectangles
            merged_rects = []
            for i in range(len(rects)):
                if len(overlapping_dict[i]) == 0:
                    merged_rects.append(rects[i])
                else:
                    x = rects[i][0]
                    y = rects[i][1]
                    w = rects[i][2]
                    h = rects[i][3]
                    for j in overlapping_dict[i]:
                        x = min(x, rects[j][0])
                        y = min(y, rects[j][1])
                        w = max(w, rects[j][0] + rects[j][2] - x)
                        h = max(h, rects[j][1] + rects[j][3] - y)
                    merged_rects.append((x, y, w, h))
            merged_rects = list(set(merged_rects))
            final_rects = merged_rects
            # if rectangle is fully inside another rectangle remove it
            indices_to_remove = []
            for i in range(len(final_rects)):
                for j in range(len(final_rects)):
                    if i == j:
                        continue
                    if final_rects[i][0] >= final_rects[j][0] and final_rects[i][1] >= final_rects[j][1] and final_rects[i][0] + final_rects[i][2] <= final_rects[j][0] + final_rects[j][2] and final_rects[i][1] + final_rects[i][3] <= final_rects[j][1] + final_rects[j][3]:
                        indices_to_remove.append(i)

            final_rects = [final_rects[i] for i in range(len(final_rects)) if i not in indices_to_remove]



        # draw the final rectangles
        for x, y, w, h in final_rects:
            cv2.rectangle(G, (x, y), (x + w, y + h), 255, 2)

        cv2.imshow("IR", G)

        if cv2.waitKey(1) and 0xFF == ord("q"):
            break
    cap.release()

first_task()

