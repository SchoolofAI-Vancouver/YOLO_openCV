# YOLO_openCV
Starter files for the Object Detection Workshop. The objective is to implement YOLOv3 using OpenCV to detect objects in an image or video.
This was repo contains the material that was covered during the Vancouver School of AI's Object Detection Workshop. The Workshop's slides can be viewed [here](https://docs.google.com/presentation/d/1Z4Pvyp2DMZ-go_SOcDDZNVFwn4TnKNq9k2iFsfGjk2k).

### Getting Started
1. Make sure you have [Python 3.6](https://www.python.org/) installed.

2. Clone the repository
    ```bash
    git clone https://github.com/mezzX/YOLO_OpenCV.git
    ```
    
3. Use [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to create a new environment and install dependencies. <br>[Click Here](https://nbviewer.jupyter.org/github/johannesgiorgis/school_of_ai_vancouver/blob/master/intro_to_data_science_tools/01_introduction_to_conda_and_jupyter_notebooks.ipynb) if you need a detail guide on using conda.

    - __Linux__ or __Mac__: 
    ```bash
    conda create --name connect4 python=3.6
    source activate connect4
    conda install numpy
    conda install opencv
    conda install jupyter notebook
    ```
  
    - __Windows__: 
    ```bash
    conda create --name connect4 python=3.6 
    activate connect4
    conda install numpy
    conda install opencv
    conda install jupyter notebook
    ```

4. Download the [YOLOv3-416 weights file](https://pjreddie.com/darknet/yolo/) and place it in the project directory


### Instructions
There are two different versions included in this repo. The .ipybn files are meant to be imported into google colab.
The .py file is meant to be ran on a local machine. To run yolo_detector.py first navigate to the directory in the terminal and run one of the three following commands

1. To run yolo_detector.py on a single image, use the following command:

    ```bash
    python yolo_detector.py --image 'image_path'
    ```

2. To run yolo_detector.py on a single video, use the following command:

    ```bash
    python yolo_detector.py --video 'video_path'
    ```
    
3. To run yolo_detector.py on a webcam feed, use the following command:

    ```bash
    python yolo_detector.py
    ```

