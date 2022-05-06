## Its
ITS and EBS related source codes

## Features
- object detection
- object ratcing
- roadway lane detection [ref](https://github.com/voldemortX/pytorch-auto-drive.git)
- vehicle lane detection
- Oracle uploading

## Get Start
Clone the source code with submodule
`git clone --recursive https://github.com/CoomaQin/its.git`

Follow the [intruction](https://github.com/voldemortX/pytorch-auto-drive.git) to download the weight of lane detection models 

Modify the visualization function and run it to generate lane keypoints. In *pytorch-auto-drive/utils/runners/lane_det_visualizer.py*, check the run function of Class LaneDetVideo to 
```
    def run(self):
        # Must do inference
        output = []
        for imgs, original_imgs in tqdm(self.dataloader):
            keypoints = None
            cps = None
            if self._cfg['pred']:
                imgs = imgs.to(self.device)
                original_imgs = original_imgs.to(self.device)
                cps, keypoints = self.lane_inference(imgs, original_imgs.shape[2:])
            results = lane_detection_visualize_batched(original_imgs,
                                                       masks=None,
                                                       keypoints=keypoints,
                                                       control_points=cps,
                                                       mask_colors=None,
                                                       keypoint_color=self._cfg['keypoint_color'],
                                                       std=None, mean=None, style=self._cfg['style'])
            results = results[..., [2, 1, 0]]
            output.append(keypoints[0])
            np.save("./kp", output)

            for j in range(results.shape[0]):
                self.writer.write(results[j])
```

Run the object detection and tracking `python3 ./object_detection_tracking.py`. Note that you need to modify the video_path, YOLO weight etc.

Fuse the lane keypoints and object bounding boxes + trakcingID to determine the vihecle position by lane `python3 ./determine_vehicle_lane.py`. Note that you need to specify the path of the above output. 

Upload results to Oracle database ``