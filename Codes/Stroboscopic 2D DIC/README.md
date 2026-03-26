# Stroboscopic 2D DIC

该目录实现了面向 SWIM 项目的频闪二维数字图像相关法（Stroboscopic 2D DIC）Python 工程。代码重点围绕以下目标设计：

- 使用普通工业相机记录与 200 Hz 超声调制严格同步的频闪视频
- 保存视频、原始帧与采集元数据
- 通过全局配准、参考区校正、公共模态抑制与局部 DIC，尽量压制手部轻微平移、旋转、姿态漂移与肌肉微颤
- 在不制作散斑的前提下，优先利用皮肤自然纹理进行相关跟踪
- 输出适合 SWIM 场景的位移场、应变场、中心位移轨迹、参考区校正轨迹、波幅剖面与叠加可视化视频

## 目录结构

- `dic/config.py`：配置模型与 YAML 加载
- `dic/capture.py`：视频录制与元数据保存
- `dic/preprocess.py`：去畸变、CLAHE、空间带通、全局 ECC 配准、参考区运动估计、公共模态抑制
- `dic/dic_core.py`：网格化局部相关、GPU/CPU 加速、亚像素位移估计与运行诊断
- `dic/analysis.py`：位移场统计、动态剪切应变、中心轨迹、XT 时空图与波前快照分析
- `dic/visualization.py`：热图、剖面、中心轨迹、XT 图、波前快照、矢量场与叠加视频输出
- `dic/cli.py`：命令行入口
- `dic/gui_common.py`：基于 `customtkinter` 的共享 GUI 组件、画布与交互工具

## 安装

在当前目录下执行：

```bash
pip install -e .
```

## 快速开始

### 1. 生成配置模板

```bash
swim-dic init-config config/example.yaml
```

### 2. 修改配置

请至少检查：

- `camera.fps` 是否与实际相机输出一致，当前默认 `60`
- `camera.disable_auto_exposure` 建议保持 `true`，并将 `camera.exposure_us` 设为接近单帧时长上限的手动曝光值，例如 `15500`
- `strobe.wave_frequency_hz = 200.0`
- `strobe.strobe_frequency_hz = 49.75`
- `strobe.pulse_width_us = 10.0`
- `analysis.pure_bright_video_fps` 建议设为 `49.75`，让亮帧拼接后的视频时间轴与闪光频率一致
- `analysis.pixel_size_um` 是否匹配你的光学倍率标定
- `dic.roi` 与 `reference_regions` 是否准确覆盖兴趣区与非兴趣参考区

### 2.1 是否需要相机标定

对于你的频闪 2D DIC 场景，建议做两层区分：

1. **如果你只关心像素级相对位移场形状**，并且只在画面中心小视场内工作，严格来说可以先不做完整畸变标定，只做一次平面尺度标定，把 `analysis.pixel_size_um` 设准。
2. **如果你要报告微米级位移、应变、波幅剖面，或者 ROI 不在光轴中心**，则建议做完整相机标定。原因是镜头畸变会把像素间距变成空间位置相关量，直接影响位移换算、剖面长度和应变估计。

你的 9×12 棋盘格、单格 15 mm 已经足够用于当前系统标定。当前工程已加入 `swim-dic calibrate` 命令，可以同时完成：

- 棋盘格角点检测
- 相机内参与畸变估计
- 自动计算平均 `pixel_size_um`
- 将结果写回 YAML 的 `analysis.pixel_size_um` 与 `calibration` 段

### 2.2 标定图像采集建议

请单独拍摄一组静态棋盘格图像，而不是从手部实验视频里截取。建议：

- 保持与正式实验**相同镜头、相同焦距、相同工作距离、相同分辨率**
- 如果实验时会固定曝光/增益，标定时也尽量保持一致
- 采集 `10` 到 `20` 张图像
- 棋盘格应覆盖画面不同位置，包含中心、四角、边缘
- 每张图改变一定姿态：平移、轻微旋转、俯仰、远近变化
- 保证棋盘格清晰对焦，避免运动模糊与过曝

如果你最终只在一个很小的中心 ROI 内做 DIC，像素尺寸也可以用该 ROI 附近的几张正视图交叉验证。

### 2.3 交互式标定图像采集

当前工程已加入一个交互式标定采集窗口。你可以实时看到摄像头画面、棋盘格角点检测结果、覆盖度、清晰度、曝光评分和总分。只有当总分高于阈值时，点击按钮或按空格才会保存图像。

如果你只想先采集图像到 `calibration_images/`：

```bash
swim-dic capture-calibration config/example.yaml calibration_images --rows 8 --cols 11 --square-mm 15.0 --min-score 70
```

窗口使用说明：

- 现在使用基于 `customtkinter` 的深色控制台式界面，而不是 OpenCV 状态面板弹窗
- 左侧实时显示相机预览与棋盘格角点覆盖结果
- 右侧面板持续显示 `Score`、`Threshold`、覆盖度、清晰度、曝光和已保存文件列表
- 点击界面中的 `Capture best frame [Space]` 按钮，或按空格，可以尝试保存当前帧
- 当未检测到完整网格，或者评分低于阈值时，程序会弹出明确原因提示并拒绝保存
- 点击 `完成并退出`，或按 `Q` / `ESC` 退出采集窗口

评分逻辑主要综合：

- 棋盘格在画面中的覆盖范围
- 图像清晰度
- 曝光是否过暗、过亮或存在明显饱和

### 2.4 标定命令

假设标定图像放在 `calibration_images/` 中，可执行：

```bash
swim-dic calibrate config/example.yaml calibration_images --rows 8 --cols 11 --square-mm 15.0 --output-json outputs/calibration/camera_calibration.json
```

该命令会：

- 读取棋盘格图像
- 用 OpenCV 检测 `8 × 11` 内角点
- 计算内参矩阵和畸变系数
- 根据相邻角点平均间距自动换算 `mm/pixel` 与 `um/pixel`
- 将结果写入 `analysis.pixel_size_um` 和 `calibration`
- 可选输出 `camera_calibration.json` 供后处理复用

如果你想一边采集一边完成标定，可以直接执行：

```bash
swim-dic calibrate-interactive config/example.yaml calibration_images --rows 8 --cols 11 --square-mm 15.0 --min-score 70 --output-json outputs/calibration/camera_calibration.json
```

如果你只想试算，不改 YAML：

```bash
swim-dic calibrate config/example.yaml calibration_images --rows 8 --cols 11 --square-mm 15.0 --no-write-config
```

### 2.5 标定后保存的数据

标定完成后，配置文件会自动保存：

- `analysis.pixel_size_um`
- `calibration.camera_matrix`
- `calibration.distortion_coefficients`
- `calibration.optimal_camera_matrix`
- `calibration.roi`
- `calibration.mean_reprojection_error_px`
- `calibration.pixel_size_std_um`

其中最直接影响后续 DIC 定量分析的是 `analysis.pixel_size_um`。当前 [`analysis.py`](Codes/Stroboscopic 2D DIC/dic/analysis.py) 会直接使用它把像素位移转换成微米位移，并进一步计算应变和波幅剖面。现在只要 `calibration.enabled: true` 且配置中存在 `camera_matrix` 与 `distortion_coefficients`，[`preprocess.py`](Codes/Stroboscopic 2D DIC/dic/preprocess.py) 会在进入 ROI 与 DIC 之前自动执行逐帧去畸变。

### 3. 交互式配置 ROI 与参考区

当前工程已加入一个交互式 ROI 配置窗口。它会优先读取 `paths.raw_video` 指向的视频，并显示其中一帧；如果视频不存在，则退回到实时相机预览。你可以直接在图像上拖拽矩形，完成 `dic.roi` 和 `reference_regions` 的可视化配置，并将结果直接写回 YAML。

```bash
swim-dic configure-roi config/example.yaml
```

如果你想使用视频中的特定帧作为配置底图，可以额外指定：

```bash
swim-dic configure-roi config/example.yaml --frame-index 10
```

这个配置器至少包含以下关键功能：

- 基于 `customtkinter` 的双栏 GUI，左侧为图像视图，右侧为属性与控制面板
- 单帧可视化预览，优先使用实验视频帧，避免盲填像素坐标
- ROI 与多个 `reference_regions` 的分色叠加显示，降低配错风险
- 鼠标直接拖拽移动和缩放矩形框，适合快速微调
- 支持连续添加、删除、清空参考区，便于尝试不同整体运动校正方案
- 实时显示 ROI 坐标、参考区数量、当前选中对象与完整参考区列表，便于核查
- 一键保存回 YAML，保持分析流程仍然使用现有 `dic.roi` 和 `reference_regions` 字段

窗口操作说明：

- 左侧画布显示预览帧与彩色叠加框，右侧面板显示元数据、参考区列表和快捷操作按钮
- 鼠标拖动矩形内部可移动当前 ROI 或参考区
- 拖动白色控制点可缩放当前矩形
- 双击空白区域可快速在该位置生成默认 ROI
- 按 `A` 新增参考区，按 `TAB` 在 ROI 与参考区之间切换选中对象
- 按 `R` 选中 ROI
- 按 `D` 或 `Delete` 删除当前选中的参考区
- 按 `C` 清空所有参考区
- 按 `I/J/K/L` 微调当前选中区域的位置
- 按 `Shift+I/J/K/L` 微调当前选中区域的尺寸
- 按 `S` 或回车保存配置并写回 YAML
- 按 `Q` 或 `ESC` 退出且不保存

### 4. 仅录制视频

```bash
swim-dic capture config/example.yaml --duration 2.0
```

### 5. 分析已录制视频

```bash
swim-dic analyze config/example.yaml
```

执行分析时，当前版本会先自动读取 `paths.raw_video` 指向的原始 60 Hz 视频，基于逐帧平均亮度自动筛选亮帧，拼接并写出 `paths.bright_video` 指向的纯亮视频，然后再将这个纯亮视频对应的帧序列送入后续去畸变、ROI、DIC、分析和可视化流程。默认情况下，纯亮视频帧率取 `analysis.pure_bright_video_fps`；若未显式设置，则回退到 `strobe.strobe_frequency_hz`。

### 6. 一步完成录制与分析

```bash
swim-dic run config/example.yaml --duration 2.0
```

## 运动伪影抑制策略

针对你提到的 1 秒左右等效采样窗口内手部不可避免的小幅移动、转动与轻微震颤，当前管线采用多层抑制：

1. **全局 Euclidean ECC 配准**
   - 先对整帧估计刚体近似运动，抑制整体平移与小角度旋转。
2. **非兴趣区参考块校正**
   - 在 `reference_regions` 中定义不受焦点表面波影响、但与手部共同运动的皮肤区域。
   - 使用模板匹配追踪这些参考区，并将其平均运动从每个 DIC 向量中扣除。
3. **公共模态信号扣除**
   - 对整帧平均亮度/整体低频漂移做 detrend 与公共模态移除，减小 LED 波动、相机增益波动和大范围背景变化影响。
4. **局部子区相关 + 亚像素相位细化**
   - 先做模板相关，再用局部相位相关细化子像素位移。
5. **时间带通分析**
   - 在慢时间轴上对位移场做带通，突出 1 Hz 等效拍频附近的结构，压制超慢漂移和高频噪声。

这一组合非常适合你的场景：目标是分离由超声焦点调制产生的微弱皮肤表面波，而不是保留整只手的宏观运动。

## 重点输出内容

输出目录默认在 `outputs/`，包含：

- `reference_frame.png`
- `amplitude_total.png`
- `rms_u.png`, `rms_v.png`
- `temporal_std.png`
- `phase_map.png`
- `strain_xx.png`, `strain_yy.png`, `strain_xy.png`
- `center_trace.png`
- `center_trace_ref_corrected.png`
- `wave_profile.png`
- `vector_field_overlay.png`
- `dic_overlay.mp4`
- `center_trace.csv`
- `dic_results.npz`

这些图覆盖了你最可能关心的内容：

- 表面波在兴趣区的空间幅值分布
- 水平/垂直位移分量的 RMS
- 参考校正前后中心点位移对比
- 波传播方向上的振幅剖面
- 主相位分布，用于观察传播连贯性
- 简单应变场估计，用于追踪局部剪切与拉伸模式
- 原视频叠加的位移箭头场，便于直观核验 DIC 是否工作正常

当前版本另外新增：

- `reference_regions_overlay.png`：把参考区直接标在参考帧上，避免参考区配置错位
- `xt_displacement.png`：沿主传播轴的位移时空图，用于观察慢动作表面波条纹
- `xt_strain_xy.png`：沿主传播轴的剪切应变时空图，更适合看 SWIM 的传播方向与相位连续性
- `wavefront_displacement_snapshots.png`：单个慢时间周期内若干关键相位的位移场快照
- `wavefront_strain_xy_snapshots.png`：单个慢时间周期内若干关键相位的剪切应变场快照
- `strain_overlay.mp4`：将动态剪切应变场叠加到原始视频上的补充视频版本
- `runtime_diagnostics.csv`：记录网格点数、总匹配次数、GPU 是否实际启用、batch size 等耗时评估信息

如果你的目标是比较 SWIM 的五种激励方式，这一组新增输出比单纯的振幅热图更有判别力，因为它们能直接显示：

- 波前是否连续单向推进
- 是否存在明显的传播斜率
- 是否存在方向翻转后的结构重置
- 剪切应变是否集中在传播前沿而不是仅仅停留在局部幅值热点

## 示例配置

```yaml
camera:
  camera_index: 0
  width: 1280
  height: 800
  fps: 49.75
  disable_auto_exposure: true
  exposure_us: 15500.0
  gain: null
  codec: mp4v
  color: true
  backend: null
strobe:
  wave_frequency_hz: 200.0
  strobe_frequency_hz: 49.75
  target_beat_hz: 1.0
  pulse_width_us: 10.0
  led_color: green
dic:
  roi: [200, 120, 700, 500]
  subset_size_px: 31
  step_size_px: 8
  search_radius_px: 12
  gaussian_sigma_px: 0.8
  highpass_sigma_px: 9.0
  clahe_clip_limit: 2.0
  clahe_tile_grid_size: 8
  bandpass_temporal_hz: [0.2, 5.0]
  global_motion_model: euclidean
  outlier_mad_scale: 4.5
  reference_strategy: median
  use_reference_region_correction: true
  use_global_motion_correction: true
  use_common_mode_subtraction: true
  median_reference_frame_count: 21
  enable_gpu: true
  gpu_backend: auto
  gpu_batch_size: 64
  numba_parallel: true
analysis:
  expected_wave_frequency_hz: 200.0
  equivalent_slow_frequency_hz: 1.0
  pure_bright_video_fps: 49.75
  export_pure_bright_video: true
  pixel_size_um: 10.0
  beat_cycles_to_analyze: 1
  smoothing_sigma_frames: 1.0
  export_video_overlays: true
  export_field_csv: true
  export_npz: true
  visualize_quiver_stride: 3
  spatial_wave_axis: auto
calibration:
  enabled: false
  board:
    inner_corners_rows: 9
    inner_corners_cols: 12
    square_size_mm: 15.0
  camera_matrix: []
  distortion_coefficients: []
  optimal_camera_matrix: []
  roi: []
  image_size: []
  mean_reprojection_error_px: null
  rms_reprojection_error_px: null
  pixel_size_um: null
  pixel_size_std_um: null
paths:
  project_root: .
  raw_video: data/raw/capture.mp4
  bright_video: data/derived/bright_only.mp4
  frames_dir: data/frames
  output_dir: outputs
  metadata_json: data/raw/capture_metadata.json
reference_regions:
  - name: ref_left
    x: 40
    y: 160
    width: 80
    height: 80
    weight: 1.0
  - name: ref_right
    x: 980
    y: 160
    width: 80
    height: 80
    weight: 1.0
notes: SWIM stroboscopic DIC measurement
```

## 参考区域配置与使用

`reference_regions` 是定义在原始相机图像坐标系中的矩形区域列表，而不是 ROI 内部坐标。也就是说，`dic.roi` 只负责告诉系统在哪里做 DIC 网格计算，而 `reference_regions` 负责告诉系统去哪里估计整只手的共同运动。

推荐规则：

- 参考区应放在与 ROI 相邻、同属手掌皮肤、但尽量不被超声焦点直接激励的位置
- 尽量放两个或以上参考区，分布在 ROI 左右或上下两侧
- 参考区内要有自然纹理，避免纯亮斑、反光区、阴影边缘
- 不要把参考区画到背景或 ROI 内强响应区域上

例如：

```yaml
reference_regions:
  - name: ref_left
    x: 40
    y: 160
    width: 80
    height: 80
    weight: 1.0
  - name: ref_right
    x: 980
    y: 160
    width: 80
    height: 80
    weight: 1.0
```

分析阶段的使用方式如下：

1. [`preprocess.py`](Codes/Stroboscopic 2D DIC/dic/preprocess.py) 先对每帧追踪各个参考区位移。
2. 然后按照 `reference_strategy` 聚合成整帧参考运动轨迹。
3. [`dic_core.py`](Codes/Stroboscopic 2D DIC/dic/dic_core.py) 在每个网格点的局部位移估计完成后，会把该参考运动从 DIC 位移中扣除。
4. [`analysis.py`](Codes/Stroboscopic 2D DIC/dic/analysis.py) 同时导出 `center_trace_ref_corrected.png`、`center_trace.csv` 与 `reference_regions_overlay.png`，用于检查参考区是否配置合理。

如果你想临时关闭参考区校正，只需把 `dic.use_reference_region_correction` 设为 `false`。

## 长耗时分析的日志与时间评估

当前版本已经为 `swim-dic analyze` 增加了面向长任务的详细日志，重点包括：

- 视频帧数、帧率、图像尺寸
- DIC 网格维度、总网格点数、subset 大小、search window 大小
- `total_matches`、每点候选搜索位置数、预计 FFT 细化次数
- GPU 是否真正启用、后端名称、batch size、Numba 并行状态
- 每帧完成时间、累计耗时、平均单帧耗时、剩余 ETA

其中 `runtime_diagnostics.csv` 会把最关键的诊断量保存下来，便于你不同配置之间横向比较。这个文件特别适合回答：为什么某次分析很慢、GPU 是否真的被启用、网格密度是否过高。

## 当前实现边界

当前版本已经是完整可运行的模块化工程，但它属于高质量研究原型，不是工业封闭式软件。后续你可以继续增强：

- 接入相机厂商 SDK，实现严格硬触发与更稳定的曝光控制
- 接入 LED 与 UMH 的 TTL 同步控制
- 用更强的逆组合 Gauss-Newton 或 FFT-DIC 替换当前基础相关器
- 加入多尺度金字塔、遮挡剔除、鲁棒外点回归
- 加入时空相位速度估计、波速拟合、群速度/相速度分析
- 对比兴趣区与非兴趣区的差分传播图，从而更直接验证 SWIM 的局部表面波增强现象
