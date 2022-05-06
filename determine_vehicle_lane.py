import numpy as np
import pandas as pd
import cv2


if __name__ == "__main__":
    # read data
    kps, ods = None, None
    with open('./data/kp.npy', 'rb') as f:
        kps = np.load(f, allow_pickle=True)
    with open('./data/od_info.npy', 'rb') as f:
        ods = np.load(f, allow_pickle=True)
    csv_path = "./Timestamp_Frame.csv"
    cnt = pd.read_csv(csv_path)
    video_path = "./its_straight.mp4"
    # print(cnt.info(verbose=True))

    # init
    COLORS = np.random.randint(0, 255, (1000, 3))
    cntidx = 0
    startx = 564
    CODEC_fourcc = "mp4v"
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(
        f'./output.mp4', cv2.VideoWriter_fourcc(*CODEC_fourcc), int(fps), (w, h))
    VIDEO = False
    right_vehicles, curr_vehicles, left_vehicles = [], [], []
    if VIDEO:
        # seperate bounding boxes to left lanes, the current lane and right lanes with respect to the vehucle where the camera mounted on
        fidx = 0
        while True:
            ret, frame = video.read()
            if ret == True:
                left, right, curr = [], [], []
                kp = kps[fidx]
                det = ods[fidx]
                centers = list(map(lambda b: ((b[2] + b[4]) // 2, (b[3] + b[5]) // 2), det))
                if len(kp) == 0:
                    print("unknown")
                    continue
                #suppose kps are sorted in dec order in a line, find the two edges of current lane, 
                left_edge, right_edge = None, None
                edge_dist = [line[0, 0] - startx for line in kp]
                right_dists = list(filter(lambda x: x > 0, edge_dist))
                left_dists = list(filter(lambda x: x < 0, edge_dist))
                right_edge = edge_dist.index(sorted(right_dists)[0]) if len(right_dists) else -1
                left_edge = edge_dist.index(sorted(left_dists)[-1]) if len(left_dists) else -1
                for eidx in [left_edge, right_edge]:
                    if eidx != -1:
                        match = [(h, 0)] * len(det)
                        for i, pt in enumerate(kp[eidx]):
                            for j, c in enumerate(centers):
                                diff = abs(c[1] - pt[1]) 
                                if diff < match[j][0]: match[j] = (diff, i) #(diff between pt_y and c_y, index of pt)
                        for j, m in enumerate(match):
                            # print(centers[j], kp[eidx][m[1]], m, eidx)
                            if centers[j][0] > kp[eidx][m[1]][0] and eidx == right_edge:
                                right.append(j)
                            elif centers[j][0] < kp[eidx][m[1]][0] and eidx == left_edge:
                                left.append(j)
                curr = list(set(range(len(det))).difference(set([*right, *left])))
                print(len(right), len(left), len(curr))
                for k, tlbr in enumerate(det):
                    label = "r" if k in right else ("l" if k in left else "c")
                    cv2.rectangle(frame, (int(tlbr[2]), int(tlbr[3])), (int(
                        tlbr[4]), int(tlbr[5])), tuple(COLORS[int(tlbr[0])].tolist()), 2)
                    cv2.putText(frame, label, (int(tlbr[2]), int(
                        tlbr[3]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, tuple(COLORS[int(tlbr[0])].tolist()), 2)
                # cv2.imshow("frame", frame)
                output.write(frame)
                cv2.waitKey(0) 
                fidx += 1
                # if fidx > 50:
                #     break
            else:
                cv2.destroyAllWindows() 
                output.release()
                break
    else:
        # seperate bounding boxes to left lanes, the current lane and right lanes with respect to the vehucle where the camera mounted on
        cntidx = 0
        while cntidx < len(cnt):
            fidx = cnt["frame"][cntidx]
            left, right, curr = [], [], []
            kp = kps[fidx]
            det = ods[fidx]
            centers = list(map(lambda b: ((b[2] + b[4]) // 2, (b[3] + b[5]) // 2), det))
            if len(kp) == 0:
                print("unknown", cntidx)
                cntidx += 1
                right_vehicles.append([])
                curr_vehicles.append([])
                left_vehicles.append([])
                continue
            #suppose kps are sorted in dec order in a line, find the two edges of current lane, 
            left_edge, right_edge = None, None
            edge_dist = [line[0, 0] - startx for line in kp]
            right_dists = list(filter(lambda x: x > 0, edge_dist))
            left_dists = list(filter(lambda x: x < 0, edge_dist))
            right_edge = edge_dist.index(sorted(right_dists)[0]) if len(right_dists) else -1
            left_edge = edge_dist.index(sorted(left_dists)[-1]) if len(left_dists) else -1
            for eidx in [left_edge, right_edge]:
                if eidx != -1:
                    match = [(h, 0)] * len(det)
                    for i, pt in enumerate(kp[eidx]):
                        for j, c in enumerate(centers):
                            diff = abs(c[1] - pt[1]) 
                            if diff < match[j][0]: match[j] = (diff, i) #(diff between pt_y and c_y, index of pt)
                    for j, m in enumerate(match):
                        # print(centers[j], kp[eidx][m[1]], m, eidx)
                        if centers[j][0] > kp[eidx][m[1]][0] and eidx == right_edge:
                            right.append(j)
                        elif centers[j][0] < kp[eidx][m[1]][0] and eidx == left_edge:
                            left.append(j)
            curr = list(set(range(len(det))).difference(set([*right, *left])))
            right_vehicles.append(right)
            curr_vehicles.append(curr)
            left_vehicles.append(left)
            cntidx += 1
        cnt["right"] = right_vehicles
        cnt["left"] = left_vehicles
        cnt["current"] = curr_vehicles
        cnt.to_csv("./its.csv")

            # if cntidx > 50:
            #     break



