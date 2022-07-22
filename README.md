# backbone_enhancement
extended from repository "p-learning", trying to implement in pytorch.

compare_weight.py: finetune.py 需要 fix 部分 layer，此部分用來確認 layer 是否真的被 fix，透過讀取訓練前後的兩個 check point，並比較各 layer 的 weights
compute_dataset_RGB_average.py: 前處理會減去 mean 除以 std，原先是使用 imagenet 的 mean, std，這邊可以根據不同 dataset 重新計算，目前是加總各圖片的平均再平均
concat_backbone.py: 當模型由兩個以上的 backbone 組成，皆透過此程式建立、儲存、讀取
configure.py: 所有模型、資料參數設定
data_argumentation.py: 這裡的 class 會用在 transforms.Compose 裡改變 image，例如 RGB 轉成 contour
extract_tar.py: 原先 imagenet dataset 是一類一個.tar，這份 code 將其解壓縮成一類一個資料夾
finetune.py: 單一 backbone 組成的模型訓練時若要 fix 部分 layer，則使用此程式達到
imagenet_example.py: 所有訓練、測試執行的程式碼，應該改名為 main.py，從 pytorch 官方的 imagenet example 修改而來
imagenet_example_origin.py: pytorch 官方的 imagenet example 原始碼
log2excel.py: 訓練過程的資訊會存在 log.txt，此程式進一步轉換，保留 train, val accuracy 方便貼到 excel 畫圖表
log_record.py: 負責記錄訓練過程的資訊，儲存在 log.txt
lr_scheduler.py: 繼承自 torch.optim.lr_scheduler.ReduceLROnPlateau 為了讓 log.txt 可以記錄 ReduceLROnPlateau 資訊
ResNet.py: 從 pytorch 官方的 torchvision.models.resnet.py 修改而來，為了要可以調整 resnet 架構
split_train_val.py: domain adaptation 的 dataset 都沒有切分 train, val，這份程式用 4:1 切分，這步驟跟一般 domain adaptation benchmark 流程不相同