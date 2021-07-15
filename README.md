###### tags: `Smart Fence`
# Smart Fence  install in Jetson nano

## Step 0 : 安裝映像檔
下載 [映像檔](https://developer.nvidia.com/embedded/dlc/jetson-nano-dev-kit-sd-card-image)，建議準備64GB或以上的記憶卡，並灌入Jetson nano

###  Environment
此映像檔環境如下
- Jetpack v43
- OpenCV 4.1 (下面教學會改安裝適合電子圍籬環境的 3.1.14 版本)
- Ubuntu 18.04
- python 3.6
- CUDA 10.0

## Step 1: 安裝虛擬環境
開機進入 Jetson nano 後，先安裝虛擬環境，用於之後的套件安裝
```bash=
sudo apt-get install virtualenv
python3 -m virtualenv -p python3 <chosen_venv_name>
source <chosen_venv_name>/bin/activate
```

## Step 2: 更新 jstson nano環境
>Smart Fence會用到Tensorflow，因此需要以下步驟安裝Tensorflow

### Install system packages required by TensorFlow

```bash=
sudo apt-get update
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
```
### Install and upgrade pip3
```bash=
sudo apt-get update
sudo apt-get install python3-pip
pip install "pip==21.1.2"
pip3 install -U pip testresources setuptools==49.6.0 
```
### Install the Python package dependencies
```bash=
sudo pip3 install -U numpy==1.19.4 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
```

### 安裝 TensorFlow GPU
Jetson Nano需要安裝 Jetpack v43的Tensorflow，才支援cuda10.0

Smart Fence的tensorflow版本需求: tensorflow>=1.14,<2.0。由於安裝1.x版本於nano有些未知問題，因此本處使用wget方式，手動下載：
```bash=
wget wget https://developer.download.nvidia.com/compute/redist/jp/v43/tensorflow/tensorflow-1.15.2+nv20.3-cp36-cp36m-linux_aarch64.whl
pip install tensorflow-1.15.2+nv20.3-cp36-cp36m-linux_aarch64.whl
```
若要安裝不同版本的tensorflow，可參考 [HERE](https://developer.download.nvidia.com/compute/redist/jp/v43/tensorflow/)。


> Reference: 
> [Jetson nano官網](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)
> [Jetson nano 網友安裝方式](https://hackmd.io/@0p3Xnj8xQ66lEl0EHA_2RQ/Skoa_phvB)
## Step 3: 下載Smart Fence
Clone 檔案

```bash=
git clone https://github.com/jim93073/SmartFence.git
cd darknet-integration
```

- 下載 [yolov4.weights](https://drive.google.com/file/d/11Hr2MAXGNerPHrGSLS0J7q0Gia73UG8U/view) 並放於 `/SmartFence/darknet-integration/libs/darknet/weights/`
- 下載 [DeepSort](https://drive.google.com/file/d/1gbrVTOa6x0F63YW80DzTVXCjar2_FaI7/view?usp=sharing)，權重，並放置於 `SmartFence/darknet-integration/libs/deep_sort/models/`

安裝Smart Fence 相關套件

>安裝Cython因為安裝scikit-learn前會用到

```bash=
pip install scikit-build==0.11.1
pip install cmake 
pip install opencv-python>=4.2.0.0
pip install numpy>=1.19.1
pip install Cython
pip install scikit-learn==0.22.1
pip install lineTool
```

如果安裝scikit-learn版本>=0.24，則須更改`SmartFence/darknet-integration/libs/deep_sort/deep_sort/`內的`linear_assignment.py`
```bash=
# from sklearn.utils.linear_assignment_ import linear_assignment # comment this line if skikit-learn >= 0.24
from scipy.optimize import linear_sum_assignment as linear_assignment # Add this line if skikit-learn >= 0.24
```



個人筆記備註 (此處無須理會) :
更改 https://hackmd.io/@U-_yXoueRIG8JDXFvXHHOA/rJtyQSDVv 內容:
1.將 `darknet-integration/libs/darknet/libyolotalk.py` 內的 "./bin/linux_x64_gpu/libyolotalk.so.1.0.1" 中拿掉 ***./*** ，修改為"bin/linux_x64_gpu/libyolotalk.so.1.0.1"
2.將requirement.txt內容修改，以符合jetson nano環境

## Step 4: Install OpenCV 3.1.14
參考 https://blog.csdn.net/ourkix/article/details/103471931

### 解除安裝內建的OpenCV 4.1
由於Smart Fence的 `Darknet-integration-in-cpp` 編譯需使用OpenCV 3 版本，因此需先解除nano內建的OpenCV

```bash=
# 查看所安裝的opencv package 
dpkg -l | grep -i opencv

# 看到相關的opencv後，就可以用apt-get來卸載
sudo apt-get remove libopencv*

# 找到所有帶opencv字符的文件並刪除
cd /usr
find . -name "*opencv*" | xargs sudo rm -rf
```





### Clone OpenCV
在 clone opencv以及opencv_contrib的時候，記得切換branch到3.4，否則可能會下載到OpenCV 4 版本
```bash=
git clone -b 3.4 https://github.com/opencv/opencv.git
git clone -b 3.4 https://github.com/opencv/opencv_contrib.git
```

### 查看cmake是否有安裝
```bash=
cmake --version 
```

沒有的話，安裝cmake
```bash=
sudo apt-get install -y cmake
```


### 安裝編譯OpenCV相關依賴
```bash=
# for ARM64
sudo add-apt-repository "deb http://ports.ubuntu.com/ubuntu-ports/ xenial-security main restricted"
sudo apt-get update
sudo apt-get install -y libjasper1 libjasper-dev
```

```bash=
sudo apt-get update
sudo apt-get install -y build-essential git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get install -y python2.7-dev python3.6-dev python-dev python-numpy python3-numpy
sudo apt-get install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
sudo apt-get install -y libv4l-dev v4l-utils qv4l2 v4l2ucp
sudo apt-get install -y curl
sudo apt-get install -y gnome gnome-devel
sudo apt-get install -y glade libglade2-dev
sudo apt-get update
```

### 編譯OpenCV
> 如果參考別的網站安裝OpenCV，cmake openCV的時候，需要設置參數: WITH_CUDA=ON 
```bash=
cd opencv
mkdir build
cd build

cmake -D WITH_CUDA=ON -D CUDA_ARCH_BIN="5.3" -D CUDA_ARCH_PTX="5.3" -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_CUDA=OFF -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules -D OPENCV_ENABLE_NONFREE=ON -D BUILD_EXAMPLES=ON ..
```

### 編譯 
```bash=
make -j4
sudo make install
```

### 確認OpenCV版本
```bash=
pkg-config --modversion opencv
```
![](https://i.imgur.com/e1ZWHi0.png)


## Step 5: Darknet-integration-in-cpp
>由於Jetson nano是AArch64(ARM64)架構，與Smart Fence 當初編譯的環境不同，因此需要重新於Jetson nano上編譯libyolotalk.so.1.0.1。

### 安裝相依套件
Clone此檔案來重新編譯libyolotalk.so.1.0.1(**需有權限**)：

```bash=
sudo apt-get install -y cmake git build-essential
```

### Clone Darknet-integration-in-cpp
```bash=
# Install libopencv-dev
sudo apt-get install libopencv-dev

# Clone this repository
git clone --recursive https://gitlab.com/lxchen-lab117/yolotalk/darknet-intergration-in-cpp.git

# Download darknet
git submodule init
git submodule update --recursive
```


### Modify file
> 由於安裝環境問題，需修改以下幾個部分

#### 修改darknet MakeFile:

將`./darknet-intergration-in-cpp/libs/darknet/MakeFile`內的GPU設為1，編譯後才會是GPU版本的libyolotalk.so.1.0.1，如下圖
![](https://i.imgur.com/4P60QCe.png =30%x)

#### 修改image_utils.cpp
因為OpenCV 3版本問題，會出現以下錯誤

![](https://i.imgur.com/rYaFike.png)

因此，將` ./darknet-intergration-in-cpp/src/image_utils.cpp` 中的92行註解，換成91行

![](https://i.imgur.com/Zbbvb04.png =50%x)

#### 修改 darknet CMakeLists.txt:
因為OpenCV 3版本問題，編譯時，會出現mavx相關錯誤
![](https://i.imgur.com/VbnbWPK.png =70%x)


將`./darknet-intergration-in-cpp/libs/darknet/CMakeLists.txt` 內的175、176註解掉

![](https://i.imgur.com/qrhifiO.png)


### Using CMake with build.sh
接著執行以下指令，會產生 `libyolotalk.so.1.0.1`檔案
```bash=
cd darknet-intergration-in-cpp
bash build.sh
```

### 複製 .so 檔案
將 `darknet-intergration-in-cpp/builds/main`裡面的 libyolotalk.so.1.0.1 移動至 Smart Fence
```bash=
cd ~/darknet-intergration-in-cpp
cp builds/libyolotalk.so.1.0.1 ~/SmartFence/darknet-integration/libs/darknet/bin/linux_x64_gpu/
```

## Step 6: 測試 Smart Fence
回到Smart Fence資料夾，運行`main.py`
```bash=
cd ~/SmartFence/darknet-integration/
python main.py
```
若成功，則會出現此畫面
![](https://i.imgur.com/UBiqn6D.png =50%x)



## Step 7: 設置IoTtalk與LineBot
設置IoTTalk與LineBot的連結

### IoTtalk setting

#### IDF
One IDF includes 4 variables:
- **object_id:** yolo device can track the people after detecting, if not using deep SORT tracking, object_id = None.
- **coordinate_x:** x coordinate of the detected object.
- **coordinate_y:** y coordinate of the detected object.
- **pointer to yolo device DB:** when detecting object, yolo device will save the frame in the path that user set, if user set None than snapshot_path = None.

Take yPerson-I as an example:
![](https://i.imgur.com/MpVL2iw.png =50%x)

#### Notice
- The object type should be set according to the image above.
- Yolo device has 80 IDFs, the same as COCO dataset’s object categories.
- IDFs’ names should follow [yolo_device IDF](https://drive.google.com/file/d/1V3VYSmKuQdXnLUxvydBkXadJdnUr6-6W/view).
- Yolo device uses market1501.pb, which can only track the person, if you want to track other objects, you can download other pb file by yourself.

#### Model
Chose the IDF you need, you can detect more than one object ( take yPerson-I for example ).

![](https://i.imgur.com/3Px2k1N.png =70%x)

#### GUI connection
IoTtalk GUI connection example:

![](https://i.imgur.com/ot3qVwn.png =70%x)

You can change DummyDevice to other devices you want to connect with yolo_device( switch set, fountain, etc ).

### LINE Notify

1. Use your LINE account to sign in LINE [Notify](https://notify-bot.line.me/zh_TW/).

![](https://i.imgur.com/04BOvQ3.png =50%x)

點選 Generate token

![](https://i.imgur.com/sEGKAkA.png =50%x)

2. Set token name and chose the chat group which will get message from LINE Notify.
3. Make sure to remember the access token.
4. `cd SmartFence/darknet-integration`
5. `vim LineNotify.py`
past the token in `token_key = ''`
6. Remember to invite LINE Notify to join your group.

![](https://i.imgur.com/m7UcJnS.png =50%x)

## Step 8: Start up Smart Fence

1. vim `SmartFence.py`
2. Smart fence doesn’t need object tracking, so we comment out line4-5:
```bash=
#from libs.deep_sort.wrapper import DeepSortWrapper
#from libs.deep_sort.wrapper import DEEP_SORT_ENCODER_MODEL_PATH_PERSON
```
3. Set the `ServerURL` and `Reg_addr` at `line15-16`. (ex:ServerURL='https://1.iottalk.tw', Reg_addr='454654') It will be wrong as follow if ServerURL and Reg_addr parameters not setup well. 

![](https://i.imgur.com/lqupGnH.png =50%x)

4. Set the device name at line20.
5. Set `video_url = 'Your camera streaming'` and the `output directory`.
6. `line72` control the sending frequency of LINE Notify.
7. Change the message of `line73` by yourself.
8. Smart fence doesn’t need object tracking, so we comment out line83-84:
```bash=
#yolo.enable_tracking(True)
#yolo.add_deep_sort_tracker("person", DeepSortWrapper(encoder_model_path=DEEP_SORT_ENCODER_MODEL_PATH_PERSON))
```
9. `python SmartFence.py`

- After successfully start the smart fence, you will get the message from LINE Notify when people entering your fence.

![](https://i.imgur.com/7kp7XRL.png =50%x)


- 完成圖
![](https://i.imgur.com/kKa39F7.jpg =50%x)










## :bulb: TroubeShotting about compile Darknet-integration-in-cpp

### 1.build時找不到OpenCV
`/usr/local/lib/cmake/opencv4/`的`OpenCVConfig.cmake`檔案，內容是指向OpenCV 4 的版本

#### 解決方法: 複製OpenCV 3.1.14 的`OpenCVConfig.cmake`
可能原本是 OpenCV 4.1 版本，後來 裝OpenCV 3.1.14 的問題。
將`opencv/release/`內的`OpenCVConfig.cmake`複製到`/usr/local/lib/cmake/opencv4`下(原本的檔名可以rename來backup以防萬一)

### 2.build時出現undefined的Error:

![](https://i.imgur.com/PUFfuXh.png =70%x)


#### 解決方法: 修改依賴
可能原本是OpenCV 4.1 版本，後來裝OpenCV 3.1.14 的問題。
將`darknet-intergration-in-cpp`下的`CMakeLists.txt`中的`find_package( OpenCV REQUIRED )` 修改為
`find_package( OpenCV 3.4.14 REQUIRED )`，即可完成編譯。







