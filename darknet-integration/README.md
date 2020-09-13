# Darknet Integration

這個Repository是一個Python的library，負責呼叫基於Darknet的C++程式，使用YOLO來辨識影片。

from: https://gitlab.com/lxchen-lab117/yolotalk/darknet-integration/

## A. 安裝

這個部分將說明如何在您的專案使用本函式庫。

1. 使用Linux系列的作業系統
2. 安裝Python 3.6以上版本
3. 將這個Repository作為git的submodule

    ```bash
    # 前往您的專案
    $ cd path/to/your/projects
    # 請先使用Git管理您的專案
    $ git init
    # 建立libs資料夾
    $ mkdir -p ./libs
    # 將libs資料夾作為python module
    $ echo "" > libs/__init__.py
    # 下載本專案
    $ git submodule add https://gitlab.com/lxchen-lab117/yolotalk/darknet-integration.git libs/darknet
    # 安裝相依套件
    $ cd libs/darknet
    $ pip install -r requirements.txt
    ```

## B. 使用
    
完整的使用範例可以參見main.py。下面是簡單的說明 :

### 1. import

如果您有按照步驟A安裝本函示庫，則本函式庫應該會位於`libs/darknet`底下。使用以下程式碼來import本函式庫。

```python
from libs.darknet import YoloDevice
from libs.darknet import DeepSortWrapper
from libs.darknet import wrapper
from libs.darknet import utils
``` 

### 2. YoloDevice物件

接著使用本函式庫的YoloDevice物件來呼叫darknet程式中的YOLO模型。
YoloDevice物件可以藉由影片網址讀入影片，並且呼叫Darknet程式來辨識影片中的物件。您可以透過以下程式碼呼叫YoloDevice物件。

```python
yolo = YoloDevice(
    video_url="rtsp://iottalk:iottalk2019@140.113.237.220:554/live2.sdp",
    gpu=False,
    gpu_id=1,
    display_message=True,
    config_file=utils.CONFIG_FILE_YOLO_V4,
    names_file=utils.NAMES_COCO,
    weights_file=utils.WEIGHTS_YOLO_V4_COCO,
    thresh=0.25,
    output_dir="./output",
    use_polygon=True,
    vertex=[(164, 0), (305, 0), (305, 90), (164, 90)],
    target_classes=["person"],
    draw_bbox=True,
    draw_polygon=True,
)
```

其中YOLO物件建構子的參數說明如下:

- `video_url`: 影片網址。
- `gpu`: 是否啟用GPU加速。
- `gpu_id`: 要使用哪一個顯示卡來加速。如果只有一張顯示卡，則填寫0。
- `display_message`: 是否印出Darknet狀態。
- `config_file`: 定義YOLO模型的設定檔的路徑。預設使用 `yolov4.cfg`。
- `names_file`: 描述標籤名稱的檔案。 預設使用 `coco.names`。
- `weights_file`: YOLO模型的權重檔案路徑。 預設使用 `yolov4.weights`。
- `thresh`: YOLO模型的門檻值(threshold)。如果物件的信心程度(confidence)小於本數值，則忽略該物件。
- `output_dir`: 影像儲存路徑。如果不使用可以填寫`None`。
- `use_polygon`: 是否指定偵測區塊(多邊形)。不使用填寫`False`。
- `vertex`: 如果`use_polygon`為`True`，則可以透過這個選項來指定多邊形的頂點。請注意，指定多邊形頂點的順序應該是逆時針的。
- `target_classes` : 指定關心的物件類別。在此清單以外的物件類別都會被忽略。不指定代表不忽略任何類別。
- `draw_bbox` : 輸出圖片時是否繪製各物件的Bounding Box。
- `draw_polygon` : 輸出圖片時是否繪製偵測區塊(多邊形)。

### 3. 定義Callback function接收回傳的物件座標

接下來要定義一個function，每當YOLO模型有偵測到物件，就會呼叫這個function來回傳物件座標。

```python
def on_data(frame_id, img, bboxes, img_path):
    """
    When objects are detected, this function will be called.

    Args:
        frame_id:
            the frame number
        img:
            a numpy array which stored the frame
        bboxes:
            A list contains several `libs.darknet.yolo_device.ExtendedBoundingBox` object. The list contains
            all detected objects in a single frame.
        img_path:
            The path of the stored frame. If `output_dir` is None, this parameter will be None too.
    """
    for det in bboxes:
        # You can push these variables to IoTtalk sever
        class_name = det.get_class_name()
        confidence = det.get_confidence()
        center_x, center_y = det.get_center()
        print(class_name, confidence, center_x, center_y, det.get_class_id(), det.get_obj_id())
```

這個function的`detections`參數是一個`list`，代表著影片的一張影格(frame)裡面的所有bounding box。
`detections`的每個元素都是一個BoundingBox物件，這個物件的詳細定義可以參見`libs.darknet.YOLO.BoundingBox`。

### 4. 綁定Callback function

定義好Callback function之後，要告訴YOLO物件如何呼叫它。我們可以透過以下程式來綁定剛剛定義好的Callback function。

```python
yolo.set_listener(on_data)
```

### 5. 使用Deep SORT追蹤物件

YoloDevice物件提供插入Deep SORT的功能。如果要使用Deep SORT，需要先建立一個tracker。

```python
tracker = DeepSortWrapper(encoder_model_path=wrapper.DEEP_SORT_ENCODER_MODEL_PATH_PERSON)
```

這個tracker主要需要輸入一個Encoder模型的路徑(`encoder_model_path`)，目前本函式庫僅提供行人追蹤的模型(`wrapper.DEEP_SORT_ENCODER_MODEL_PATH_PERSON`)。

接著啟用YoloDevice的物件追蹤功能，並將這個tracker加入YoloDevice當中。

```python
yolo.enable_tracking(True)
yolo.add_deep_sort_tracker("person", tracker)
```

請注意，`yolo.add_deep_sort_tracker`需要指定兩個參數，第一個參數是這個tracker能夠追蹤的物件類別的名稱，第二個參數是剛剛宣告的tracker。第一個參數的物件類別名稱只能從`names_file`的內容裡面挑選。

### 6. 啟動YOLO模型

設定好上述參數後，就可以啟動Darknet程式當中的YOLO模型開始辨識物件了。

```python
yolo.start()
```

請注意，上述程式會開啟一個新的執行緒來執行Darknet程式。因此，請在該行程式下方加上`while True`迴圈，或使用其他方式避免主執行緒結束。

### 7. 關閉YOLO模型

您可以使用以下程式碼來結束Darknet程式。這將會關閉運行Darknet程式的執行緒。

```python
yolo.stop()
```

和一般多執行緒程式寫法一樣，您可以透過以下程式來等待執行緒結束。

```python
yolo.join()
```

## C. 說明

我將darknet做為C++的library，將darknet裡面YOLO prediction的部分另外獨立寫成了一支程式。這麼做主要是為了避免修改darknet的程式碼。
這個Repository的python程式實際上呼叫的是我新寫的C++程式。關於這個C++程式的細節可以參見: [https://gitlab.com/lxchen-lab117/yolotalk/darknet-integration-in-cpp](https://gitlab.com/lxchen-lab117/yolotalk/darknet-integration-in-cpp)。
