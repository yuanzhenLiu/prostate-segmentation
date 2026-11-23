# 基于 Classic U-Net 的视频抽帧图像二值分割

本仓库整理了最终 **Classic U-Net v2** 的完整流程，包括患者级数据划分、训练集连续帧抽样、GPU训练、测试集评价、单图或文件夹推理以及训练曲线绘制。

真实医学图像、mask、患者级划分表、训练权重和病例可视化均为私有数据，不包含在仓库中。

## 最终方法

- `image1`对应`mask1`，`image2`对应`mask2`。
- 按患者约7:1:2划分训练、验证和测试集，同一患者不会跨集合。
- 只对训练集抽帧：按来源和患者分组，按帧号排序，每4帧取1帧，每组最多保留前180个抽样帧；验证集和测试集不抽帧。
- Classic U-Net从头训练：4层编码器、4层解码器，基础通道24，dropout 0.15，单通道二值输出，无预训练backbone。
- 训练和测试评价统一缩放到512×512。
- AdamW，学习率`1e-4`，weight decay `3e-3`，BCE+soft Dice损失，batch size 8，最多35轮，早停patience 5。
- 数据增强包括水平翻转、±10°旋转、轻微亮度和对比度、Gaussian噪声及模糊；不进行上下翻转。
- 新图推理先在512×512计算概率，再恢复到原图尺寸；ROI使用红色`(255,0,0)`和0.45透明度融合，只改变ROI区域。

准确参数见[`configs/unet_v2.json`](configs/unet_v2.json)，实现细节见[`docs/METHOD.md`](docs/METHOD.md)。

## 安装环境

推荐Python 3.10或以上版本。先按照新机器的显卡和驱动安装合适的GPU版PyTorch，再安装其余依赖。`requirements.txt`没有包含PyTorch，避免覆盖已经可以正常使用的CUDA环境。

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print((torch.ones(1, device='cuda') + 1).item())"
```

最后一条命令必须正常结束才能进行GPU训练。推理也支持`--device cpu`。

## 数据目录和命名

```text
DATA_ROOT/
├── image1/
├── mask1/
├── image2/
└── mask2/
```

图像与mask文件名的stem必须完全相同。文件名格式为`P-XXX_YYY`，其中`XXX`是患者编号，`YYY`是帧号。

生成私有患者级划分表：

```powershell
python scripts/prepare_split.py --data-root "D:\path\to\date" --output "D:\private\dataset_split.xlsx" --seed 2026
```

该表含患者编号和绝对路径，不得提交到Git。数据迁移后可保留原划分，仅更新路径：

```powershell
python scripts/relocate_split.py --input "D:\private\old_split.xlsx" --data-root "E:\new\date" --output "E:\private\dataset_split.xlsx"
```

## 训练

训练必须使用CUDA：

```powershell
python scripts/train.py --config configs/unet_v2.json --split-table "D:\private\dataset_split.xlsx" --output-dir "D:\private\unet_v2_run"
```

Windows多进程加载报错时添加`--num-workers 0`。显存不足时可添加`--batch-size 4`，但这会改变运行参数。原历史实验没有完整设置全局随机种子；今后的运行可加`--seed 2026`提高可重复性，但不能声称可精确复现原数值。

## 测试集评价和可视化

```powershell
python scripts/evaluate.py --config configs/unet_v2.json --split-table "D:\private\dataset_split.xlsx" --checkpoint "D:\private\unet_best.pt" --output-dir "D:\private\evaluation" --batch-size 4 --num-visuals 30 --device cuda
```

程序会处理每个batch中的全部样本，输出逐图指标、汇总指标、红色预测融合图，以及“原图/绿色真值/红色预测”比较图。指标在512×512上计算，因此HD95单位是该网格上的像素。汇总值是逐帧均值和样本标准差，连续帧不能视为相互独立的患者。

## 输入无标签新图

单张图片：

```powershell
python scripts/predict.py --input "D:\images\example.png" --checkpoint "D:\private\unet_best.pt" --output-dir "D:\predictions" --device cuda
```

递归处理整个文件夹：

```powershell
python scripts/predict.py --input "D:\images" --checkpoint "D:\private\unet_best.pt" --output-dir "D:\predictions" --device auto
```

每张图生成`*_mask.png`、`*_overlay.png`和`*_probability.npy`，尺寸与输入原图一致。无标签新图只能得到分割结果，不能计算真实Dice等准确性指标。

## 绘制曲线

```powershell
python scripts/plot_curves.py --history "D:\private\unet_v2_run\history_unet.json" --output-dir "D:\private\unet_v2_run\curves"
```

训练曲线为实线，验证曲线为同色虚线。

## 已核实的历史结果

最终私有v2实验包含67位患者和23,106对图像。训练/验证/测试分别为47/7/13位患者及16,288/2,161/4,657帧。训练在第16轮早停，第11轮验证Dice最高，为0.825984。

测试集逐帧均值：Dice 0.823264、IoU 0.722785、Precision 0.850261、Recall/Sensitivity 0.848139、HD95 29.198682像素。这些数值对应私有历史实验；仓库没有公开数据和权重，也不保证重训后完全一致。

## 常见问题与复现边界

- CUDA不可用：安装与显卡和驱动匹配的PyTorch，然后重新运行GPU验证命令。
- `no kernel image is available`：当前PyTorch CUDA版本不支持该GPU架构。
- 找不到图像或mask：检查四个目录、同名stem及划分表路径，必要时运行路径迁移脚本。
- PowerShell不能直接运行`train.py`；应输入`python scripts/train.py ...`。
- 私有数据、划分表和权重需自行准备，详见[`checkpoints/README.md`](checkpoints/README.md)。

公开版中两处不改变训练方法的程序修复见[`docs/OUTPUTS_AND_REPRODUCIBILITY.md`](docs/OUTPUTS_AND_REPRODUCIBILITY.md)。

## 许可证

当前没有添加LICENSE。公开发布前，请根据所在机构要求、数据协议和预期用途选择许可证。本仓库没有虚构作者、机构、论文引用或公开权重链接。
