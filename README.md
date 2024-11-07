# Motion Detector
Python Application using Kalman Filters

Application currently have trouble with close-up motion, due to the parameters being set to far away objects.
Feel free the tweak the parameters, which are

- **`alpha`**: Frame hysteresis for active/inactive objects. This value determines how many frames an object must be inactive before it is considered lost.

  - **Type**: `int`
  - **Default**: `5`
  - **Description**: The number of consecutive frames an object can remain inactive (missed) before being marked as inactive. A higher value results in longer periods before an object is lost.

  - **`tau`**: Threshold for motion, in order to filter out random noise.

  - **Type**: `int`
  - **Default**: `0.2`
  - **Description**: The pixel difference threshold used to detect motion between frames. A higher value makes the detection less sensitive, while a lower value makes it more sensitive to small changes.

- **`delta`**: Maximum allowed distance for matching objects between frames.
  - **Type**: `int`
  - **Default**: `20`
  - **Description**: This distance defines how far apart the centroids of tracked objects can be before they are considered different objects. Lower values are more restrictive, and higher values allow more flexibility.

- **`s`**: Number of frames to skip between detections.
  - **Type**: `int`
  - **Default**: `1`
  - **Description**: This parameter controls how many frames are skipped between updates. With a higher frame skip, this should improve performance or reduce unncessary updates.

- **`N`**: Maximum number of objects to track.
  - **Type**: `int`
  - **Default**: `40`
  - **Description**: This is the maximum number of objects that can be tracked at any given time. Once this limit is reached, new objects will not be tracked until some of the existing objects are no longer active.


## Badges

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


## Run Locally

Clone the project, or download the two files.

```bash
  git clone https://github.com/TahMeat/Motion-Detector/
```



Go to the project directory

```bash
  cd Motion-Detector/
```

Run the program

```bash
  python .\track_objects.py [input_file]
```

#### Arguments

- `input_file`: The path to the video file.
