# HUMANOIDGEN Training

이 문서는 LeRobot 형식의 `HUMANOIDGEN_DATA`로 StarVLA `QwenGR00T`를 학습하는 방법을 설명한다.

기본값:

- dataset root: `playground/Datasets/HUMANOIDGEN_DATA`
- robot: GR1-style bimanual humanoid
- framework: `QwenGR00T`
- base VLM: `playground/Pretrained_models/Qwen3.5-2B`
- action dim: `26`
- state dim: `26`
- action horizon: `16`
- image size: `224 x 224`
- camera views: head, left wrist, right wrist

## 파일 구조

```text
examples/HUMANOIDGEN/
├── README.md
└── train_files/
    ├── run_humanoidgen_train.sh
    ├── starvla_train_humanoidgen.yaml
    └── data_registry/
        └── data_config.py
```

- `run_humanoidgen_train.sh`: 실제 학습 실행 스크립트.
- `starvla_train_humanoidgen.yaml`: 기본 학습 config.
- `data_registry/data_config.py`: `HUMANOIDGEN_DATA`의 LeRobot 컬럼을 StarVLA 입력으로 매핑.

## 데이터 준비

`HUMANOIDGEN_DATA`는 LeRobot dataset root여야 한다. 기본 실행은 아래 경로를 본다.

```bash
playground/Datasets/HUMANOIDGEN_DATA
```

다른 위치에 있으면 환경 변수로 넘긴다.

```bash
export HUMANOIDGEN_DATA=/path/to/HUMANOIDGEN_DATA
```

현재 registry는 dataset root 바로 아래를 하나의 dataset으로 쓴다. 즉 mixture는 `(".", 1.0, "humanoidgen_gr1")`이다. 여러 하위 dataset을 섞고 싶으면 `DATASET_NAMED_MIXTURES`에 각 폴더를 추가한다.

예:

```python
DATASET_NAMED_MIXTURES = {
    "humanoidgen_all": [
        ("task_a", 1.0, "humanoidgen_gr1"),
        ("task_b", 1.0, "humanoidgen_gr1"),
    ],
}
```

## LeRobot 컬럼 매핑

기본 컬럼은 `examples/HUMANOIDGEN/train_files/data_registry/data_config.py`에 있다.

```python
video_keys = [
    "video.head_camera",
    "video.left_wrist_camera",
    "video.right_wrist_camera",
]
state_keys = ["state.joints"]
action_keys = ["action.joints"]
language_keys = ["annotation.human.task_description"]
```

모델 입력 의미:

- `video.*`: 관측 이미지. 현재 step의 head/left wrist/right wrist 영상을 읽는다.
- `state.joints`: 현재 로봇 state. dim은 `26`.
- `action.joints`: 미래 action chunk. dim은 `26`.
- `annotation.human.task_description`: language instruction.

시간 index:

- `observation_indices = [0]`: 현재 frame만 observation으로 사용.
- `action_indices = list(range(16))`: 현재 step부터 16 step action chunk 학습.

state/action 정규화:

- `StateActionToTensor`로 tensor 변환.
- `StateActionTransform`에서 `min_max` 정규화.
- 학습 시작 때 dataset statistics가 계산되어 run dir에 저장된다.

언어 컬럼이 `annotation.human.coarse_action`이면 coarse mixture를 쓴다.

```bash
data_mix=humanoidgen_all_coarse \
bash examples/HUMANOIDGEN/train_files/run_humanoidgen_train.sh
```

컬럼명이 다르면 `data_config.py`의 `video_keys`, `state_keys`, `action_keys`, `language_keys`, dim 값을 데이터에 맞게 고친다.

## 기본 학습 실행

4 GPU 예:

```bash
HUMANOIDGEN_DATA=/path/to/HUMANOIDGEN_DATA \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_PROCESSES=4 \
bash examples/HUMANOIDGEN/train_files/run_humanoidgen_train.sh
```

스크립트는 내부에서 아래 명령을 만든다.

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/HUMANOIDGEN/train_files/starvla_train_humanoidgen.yaml
```

나머지 값은 shell 환경 변수로 config를 override한다.

## 자주 바꾸는 옵션

```bash
HUMANOIDGEN_DATA=/path/to/HUMANOIDGEN_DATA \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_PROCESSES=4 \
base_vlm=/path/to/Qwen3.5-2B \
run_root_dir=./playground/Checkpoints \
run_id=humanoidgen_qwengroot_exp01 \
per_device_batch_size=16 \
max_train_steps=50000 \
save_interval=5000 \
bash examples/HUMANOIDGEN/train_files/run_humanoidgen_train.sh
```

주요 override:

- `HUMANOIDGEN_DATA`: dataset root.
- `base_vlm`: Qwen VLM checkpoint path.
- `run_root_dir`: 결과 저장 root.
- `run_id`: run 이름.
- `per_device_batch_size`: GPU당 batch size.
- `max_train_steps`: 총 optimization step.
- `save_interval`: checkpoint 저장 간격.
- `data_mix`: 사용할 dataset mixture.
- `video_backend`: video reader. 기본값은 `torchcodec`.
- `action_dim`, `state_dim`, `action_horizon`: 데이터 shape과 맞춰야 하는 값.
- `freeze_module_list`: freeze할 module 이름. 기본은 empty string.

## 학습 설정 요약

`starvla_train_humanoidgen.yaml`의 핵심값:

```yaml
framework:
  name: QwenGR00T
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen3.5-2B
  action_model:
    action_model_type: DiT-B
    action_dim: 26
    state_dim: 26
    action_horizon: 16

datasets:
  vla_data:
    dataset_py: lerobot_datasets
    include_state: true
    data_root_dir: playground/Datasets/HUMANOIDGEN_DATA
    data_mix: humanoidgen_all
    action_type: delta_ee
    per_device_batch_size: 8
    load_all_data_for_training: true
    obs_image_size: [224, 224]
    video_backend: torchcodec

trainer:
  max_train_steps: 100000
  num_warmup_steps: 5000
  save_interval: 10000
  learning_rate:
    base: 2.5e-05
    qwen_vl_interface: 1.0e-05
    action_model: 1.0e-04
```

실제 실행 때 `run_humanoidgen_train.sh`가 일부 값을 CLI로 덮어쓴다. shell 변수로 넘긴 값이 YAML보다 우선한다.

## 학습 흐름

1. `accelerate launch`가 GPU 수만큼 process를 띄운다.
2. `train_starvla.py`가 YAML과 CLI override를 merge한다.
3. `build_framework`가 `QwenGR00T` 모델을 만든다.
4. `build_dataloader`가 `lerobot_datasets`를 선택한다.
5. registry가 `examples/*/train_files/data_registry/`를 자동 탐색한다.
6. `humanoidgen_all` mixture가 `HUMANOIDGEN_DATA`를 `LeRobotSingleDataset`으로 연다.
7. dataset statistics를 계산하고 `dataset_statistics.json`으로 저장한다.
8. trainer가 action DiT loss로 학습한다.
9. `eval_interval`마다 현재 batch로 action MSE를 계산한다.
10. `save_interval`마다 checkpoint를 저장한다.

## 출력물

기본 출력 경로:

```bash
playground/Checkpoints/humanoidgen_qwengroot
```

주요 파일:

- `config.full.yaml`: merge된 전체 config.
- `config.yaml`: 학습 중 실제 접근된 config snapshot.
- `dataset_statistics.json`: action/state 정규화 통계.
- `checkpoints/steps_<N>_pytorch_model.pt`: step checkpoint.
- `summary.jsonl`: 저장 step 기록.
- `final_model/pytorch_model.pt`: 마지막 모델.
- `wandb/`: W&B local log. 기본은 `WANDB_MODE=disabled`.

## Resume / pretrained load

현재 실행 스크립트에는 resume 옵션이 기본으로 노출되어 있지 않다. 필요하면 YAML 또는 CLI override에 trainer 값을 추가한다.

예:

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/HUMANOIDGEN/train_files/starvla_train_humanoidgen.yaml \
  --trainer.is_resume true
```

특정 pretrained checkpoint를 일부 module에 load하려면 `trainer.pretrained_checkpoint`, `trainer.reload_modules`를 config에 넣는다.

## 체크 포인트

- `HUMANOIDGEN_DATA` path가 맞는지 먼저 확인.
- LeRobot 컬럼명이 registry와 같은지 확인.
- `state_dim`, `action_dim`, `action_horizon`이 실제 데이터와 같은지 확인.
- `base_vlm` path에 Qwen checkpoint가 있는지 확인.
- video decode가 실패하면 `video_backend=torchvision_av` 또는 `video_backend=decord`를 시도.
- GPU memory가 부족하면 `per_device_batch_size`를 줄인다.
- multi-node가 아니면 기본 NCCL 값 그대로 사용해도 된다.
