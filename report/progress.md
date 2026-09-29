# Lab 3 Plan & Progress

- 과제 명세: [spec.md](spec.md)
- 마감: **2026-10-04 (일) 23:59 KST**
- 전략: **로컬(macOS, GPU 없음)에서 코드를 먼저 완성·정적 검증** → Vast.ai 인스턴스를 한 번만 빌려서 실행·로그 수집 → 즉시 destroy → 로컬에서 분석·보고서 작성
  - 인스턴스 사용 시간을 최소화하는 것이 목표 (과금은 instance가 존재하는 동안 계속됨)

---

## 진행 현황 (Checklist)

### Phase A. 로컬 코드 구현 (Vast.ai 불필요)

- [x] **P0** `scripts/launch.sh` — `nsys profile` 명령 추가
- [x] **P1** `handler/DDP/utils.py::run_process` — `mp.spawn`
- [x] **P2** `handler/DDP/utils.py::initialize_group` — `dist.init_process_group` + `torch.cuda.set_device`
- [x] **P3** `handler/DDP/utils.py::destroy_process` — `dist.destroy_process_group`
- [x] **P4** `handler/DDP/model.py::model_to_DDP` — `DDP(model.cuda(dev), device_ids=[dev])`
- [x] **P5** `handler/DDP/cifar10_loader.py::get_DDP_loader` — `DistributedSampler`
- [x] **P6** `handler/DALI/cifar10_loader.py::CifarPipeline` — pad/flip/crop/normalize/cutout
- [x] **P7** `handler/DALI/cifar10_loader.py::get_DALI_loader` — sharded pipeline + `DALIGenericIterator`
- [~] 로컬 정적 검증 — `py_compile` / `bash -n` 통과. 로컬에 torch가 없어 gloo 스모크 테스트는 생략 → Phase C 스모크 테스트에서 대체
- [ ] commit

### Phase C. Vast.ai 실행

- [x] 인스턴스 대여 — 2× RTX 3090 (24GB), driver 580.178 (CUDA 13.0), torch 2.4.1+cu118, nsys 2025.5.1, disk 32GB. SSH: `vast-direct` / `vast-proxy` (`~/.ssh/config`)
- [x] `bash scripts/init.sh` — 첫 시도는 다운로드가 60%에서 멈춰 재실행. train 50,000 / test 10,000 PNG, `check_env.sh` 통과 (P2P access: False)
- [x] 스모크 테스트 (2 GPU, nsys 없음) — DDP·DALI 모두 통과, 코드 수정 없음
- [x] `bash run_vastai_dp.sh` — `dp_20260929_141639`
- [x] `bash run_vastai_ddp.sh` — `ddp_20260929_141819`
- [x] `bash run_vastai_ddp_dali.sh` — `ddp_dali_20260929_141948`
- [x] `nsight_logs/`, `logs/` 다운로드 (scp, 49개 파일 MD5 일치) — 로컬 저장소의 `nsight_logs/`, `logs/` (gitignore 대상)
- [x] **instance destroy**

### Phase D. 분석 & 보고서

- [x] TA 제공 `.nsys-rep` 다운로드 (gdown) → `nsight_logs/ta/{dp,ddp,ddp_dali}/gpu_{1,2,4}.nsys-rep` (9개, RTX 3090). spec에는 DP만 준다고 되어 있었으나 DDP/DDP+DALI도 포함
- [ ] **To-do (GUI 수작업)**: TA 로그 9개를 `nsys-ui`에서 열어 **File → Export → SQLite**로 `nsight_logs/ta/{dp,ddp,ddp_dali}/gpu_{1,2,4}.sqlite`에 저장 → 그 뒤 요약 표 계산
  - Mac용 Nsight Systems에는 명령줄 `nsys stats`/`export`가 없음 (앱 안 `nsys`는 Linux용). 인스턴스는 destroy됨
  - 대안: Docker Desktop을 켜고 Linux용 `nsight-systems-cli` 컨테이너에서 `nsys stats` 실행
- [ ] **To-do (GUI 수작업)**: 보고서용 타임라인 스크린샷 (DP / DDP / DDP+DALI, 2 GPU, NVTX + CUDA 행)
- [x] 로컬 Nsight Systems 설치 — 2026.5.1 (인스턴스 2025.5.1 이상)
- [ ] 수치 추출 & 비교표 작성
- [ ] 보고서 작성 (≤ 4 pages, 11pt) → `lab3_DDP_DALI_team{N}.pdf`
- [ ] 제출 zip 생성 → PLMS 업로드

---

## Phase A. 구현 상세 계획

코드 흐름 (`train_cifar.py`):
`run_process(main_func)` → 각 프로세스에서 `initialize_group` → `get_loader(test_batch=512, train_batch=512)` → `ResNet18` → `model_to_DDP` → `sync_checkpoint` → `train/test` (1 epoch) → `destroy_process`.

### P0. `scripts/launch.sh` — Nsight 로그 생성

scaffold의 `python train_cifar.py ...` 앞에 `nsys profile`을 붙인다. 출력 경로는 run 스크립트가 export하는 `$NSIGHT_LOG_DIR/$NSIGHT_FILE_NAME` 사용.

```bash
CUDA_VISIBLE_DEVICES=$LOCAL_GPU_IDS nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --sample=none --cpuctxsw=none \
    --force-overwrite=true \
    --output="$NSIGHT_LOG_DIR/$NSIGHT_FILE_NAME" \
    python train_cifar.py \
        --num_gpu=$NUM_GPUS --data="$DATA_DIR" --ckpt="$CKPT_DIR" --mode="$MODE" --save_ckpt
```

- `--trace=...nvtx`: 코드에 있는 `nvtx.annotate` 구간(Epoch / batch / forward / backward) 기록 — 필수
- `--cuda-memory-usage=true`: GPU 메모리 사용량 타임라인 (Mem util 분석용)
- `--sample=none --cpuctxsw=none`: Vast.ai 컨테이너는 unprivileged라 CPU sampling 불가 (README Troubleshooting). 경고 방지용
- DDP는 `mp.spawn`으로 자식 프로세스를 만들지만 nsys는 기본적으로 자식 프로세스도 추적함 → 한 `.nsys-rep`에 rank 0/1 모두 기록되는지 Phase C에서 확인
- **GPU utilization (SM 활용률) 메트릭**: `--gpu-metrics-devices=all` (nsys 버전에 따라 `--gpu-metrics-device`) 은 root/권한이 필요해 Vast.ai에서 실패할 가능성이 있음 → Phase C 스모크 테스트에서 시도해 보고, 안 되면 제외하고 CUDA kernel 타임라인 기반으로 "GPU busy 비율"을 계산 (Phase D 참고)
- 선택: `--export=sqlite` 는 로컬에서 `nsys export`로도 만들 수 있으므로 생략

### P1. `run_process(func, args)`

```python
mp.spawn(func, args=(args,), nprocs=args.num_gpu, join=True)
```
- `mp.spawn`은 `func(proc_id, *args)` 형태로 호출 → `main_func(proc_id, args)` 시그니처와 일치
- 참고: spawn 방식이므로 자식 프로세스가 `train_cifar.py`를 다시 import하며 모듈 레벨 `argparse`/`find_free_port()`가 재실행되지만, `main_func`에는 부모의 `args`(pickle)가 전달되므로 port는 일관됨

### P2. `initialize_group(proc_id, host, port, num_gpu)`

```python
dist.init_process_group(backend="nccl", init_method=dist_url, world_size=num_gpu, rank=proc_id)
torch.cuda.set_device(proc_id)
```
- `CUDA_VISIBLE_DEVICES`로 GPU를 제한하므로 local rank == `proc_id` == device index
- 이후 `torch.cuda.current_device()`가 각 프로세스의 GPU를 가리키며, `sync_checkpoint`/`resume_checkpoint`의 `map_location`이 이에 의존함 → `set_device`가 반드시 필요

### P3. `destroy_process()`

```python
if dist.is_initialized():
    dist.destroy_process_group()
```
- `finally` 블록에서 호출되므로 초기화 실패 시에도 에러가 나지 않도록 `is_initialized()` 가드

### P4. `model_to_DDP(model)`

```python
device = torch.cuda.current_device()
model = model.cuda(device)
model = DDP(model, device_ids=[device], output_device=device)
```
- DP는 `device_ids=list(range(n))` (한 프로세스가 여러 GPU)인 반면, DDP는 **프로세스당 1 GPU** → `device_ids=[local_rank]`
- BatchNorm: 기본 BN (rank별 통계). SyncBatchNorm은 명세에 없으므로 사용하지 않음 (보고서에 언급 가능)

### P5. `get_DDP_loader(...)`

scaffold 주석을 채우는 방식. DP 로더 대비 변경점:

1. `world_size = dist.get_world_size()`, `rank = dist.get_rank()`
2. `DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=...)`를 만들고 DataLoader에 `sampler=`로 전달, **`shuffle` 인자는 제거** (sampler와 동시 사용 불가)
3. **배치 크기 결정 (중요)**: `train_batch // world_size` (per-GPU 배치) 로 나눠서 **global batch = 512 유지**
   - 근거: DP 로더 주석에 `assert(train_batch % world_size == 0)`가 남아 있음 → 원래 코드가 per-rank로 나누는 설계였음
   - DP와 동일한 global batch/lr 조건에서 비교해야 공정 → 보고서에서 "같은 일을 몇 GPU로 나눴을 때"를 비교 가능
   - `DDP/train.py`의 `reduce_tensor`로 batch_size/loss/acc를 합산하므로 로깅도 global 기준으로 맞음
4. test loader도 `DistributedSampler(shuffle=False)` 사용 (`test()`에서 all_reduce로 합산)
   - 주의: 10000 / 2 = 5000으로 나누어떨어져 패딩 중복 없음
5. Cutout: 기존과 동일하게 `transform_train.transforms.append(Cutout(cutout))`
6. `sampler.set_epoch(epoch)`는 `train.py`에서 호출하지 않음 — `target_epoch = 1`이라 영향 없음 (보고서에 한 줄 언급)

### P6. `CifarPipeline`

scaffold의 `__init__` 그대로 사용 (`Pipeline(batch_size, num_threads=num_workers, device_id, seed=12345)`).
torchvision 파이프라인과 **동일한 동작**이 되도록 매핑:

| torchvision (DP/DDP) | DALI |
|---|---|
| ImageFolder 읽기 | `fn.readers.file(file_root=data_dir, shard_id, num_shards, random_shuffle=is_train, pad_last_batch=True, name="Reader")` |
| (PIL decode) | `fn.decoders.image(device="mixed", output_type=RGB)` → GPU, HWC uint8 |
| `RandomCrop(32, padding=4)` | ① `fn.paste(images, ratio=1.25, fill_value=0)` 로 중앙에 두고 40×40 zero padding ② `fn.crop_mirror_normalize(crop=(32,32), crop_pos_x=U(0,1), crop_pos_y=U(0,1))` |
| `RandomHorizontalFlip()` | `mirror = fn.random.coin_flip(probability=0.5)` → CMN의 `mirror=` 인자 |
| `ToTensor() + Normalize(mean,std)` | CMN: `mean=CIFAR_MEAN, std=CIFAR_STD` (0–255 스케일, 파일 상단에 정의됨), `dtype=FLOAT`, `output_layout="CHW"` |
| `Cutout(16)` (normalize 후, 0으로 채움) | `fn_dali_cutout(images, cutout_length)` (CHW, fill 0.0) — 제공 함수 |
| test: `ToTensor + Normalize` | CMN (crop 없음 / mirror 없음) |

- 패딩 방식 대안: `fn.pad`는 끝쪽만 패딩하므로 부적합 → `fn.paste`(중앙 배치) 사용. 동작 확인은 Phase C에서 출력 shape `(B,3,32,32)` 와 값 범위로 검증
- 차이점 (보고서에 명시):
  - 제공된 `fn_dali_cutout`은 사각형이 항상 이미지 내부에 완전히 들어감 (torch 버전은 중심이 임의 위치 + clip) → 미세한 분포 차이
  - PNG는 nvJPEG 대상이 아니라 `mixed` 디코더라도 CPU에서 디코드됨 → DALI 이득은 주로 augmentation GPU화 + 비동기 prefetch
- 라벨 순서: `fn.readers.file`은 하위 폴더명을 정렬해 라벨 부여. `init_dataset.py`가 torchvision `classes` 이름(airplane…truck, 이미 알파벳 순)으로 폴더를 만들므로 라벨 일치 ✅
- seed: 모든 shard가 같은 seed(12345)를 쓰면 augmentation 난수열이 동일 → 데이터가 다르므로 치명적이진 않으나, `seed=12345 + shard_id` 로 바꾸는 것을 고려 (scaffold 변경 최소화 vs 정확성 — 구현 시 결정)

### P7. `get_DALI_loader(...)`

```python
world_size, rank = dist.get_world_size(), dist.get_rank()
device_id = torch.cuda.current_device()
pipe = CifarPipeline(train_dir, train_batch // world_size, True, cutout,
                     device_id, rank, world_size, num_workers)
pipe.build()
dali_iter = DALIGenericIterator(pipe, ["data", "label"], reader_name="Reader",
                                last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
train_loader = DALIWrapper(dali_iter)
```
- `root`는 `dataset/cifar10_images` → `train/`, `test/` 하위 폴더 사용 (`valid/`는 비어 있음, `valid_size=0`이므로 스킵)
- `reader_name="Reader"` 지정 시 iterator가 shard 크기를 자동 계산 → `DALIWrapper.__len__` 정상 동작
- `LastBatchPolicy.PARTIAL`: DP/DDP의 `drop_last=False`와 동일한 의미 (마지막 배치 작게)
- `auto_reset=True`: `DALIWrapper.__next__`가 `StopIteration`을 그대로 올리므로, 다음 epoch/재사용 시 reset 필요 → auto_reset으로 처리
- test 파이프라인: `is_train=False`, `cutout=0`, `random_shuffle=False`
- `download` 인자는 사용하지 않음 (시그니처 호환용)
- `num_workers=4` (DALI CPU 스레드 수, 함수 기본값)

---

## Phase B. 로컬 검증 (macOS, GPU 없음)

DALI/NCCL/CUDA는 로컬에서 실행 불가하므로 가능한 범위만 검증:

1. `python -m py_compile` 로 모든 수정 파일 문법 확인
2. `bash -n scripts/launch.sh` 로 쉘 문법 확인
3. (가능하면) CPU 스모크 테스트: 임시 스크립트(스크래치 디렉터리, 커밋 안 함)에서 `gloo` backend로 2 프로세스를 띄워
   - `DistributedSampler` 분할이 겹치지 않고 전체를 덮는지 (rank0 ∪ rank1 = 50000, 교집합 = ∅)
   - per-rank batch = 256 인지
   - 확인 (로컬에 CIFAR-10 torchvision 데이터가 필요 — 없으면 dummy dataset으로 대체)
4. 코드 리뷰: device id, sampler, batch 나누기, DALI 파라미터 이름을 DALI 1.x 문서와 대조

## Phase C. Vast.ai 실행 계획

**준비물**: SSH 키 등록, 이 repo를 GitHub에 push (인스턴스에서 clone) — 또는 scp로 업로드

1. **대여**: 2 GPU, RTX 3090 (TA 로그와 동일 HW → 비교 가능), Max CUDA ≥ 12.1, disk ≥ 50 GB, image `25fallcsed490f/cluster:week4`, SSH 모드
2. **셋업** (tmux 안에서):
   ```bash
   cd /workspace && git clone <my repo> && cd CSED490F_DDP_DALI_Training
   bash scripts/init.sh          # 수 분 소요 (PNG 60,000장 저장)
   bash scripts/check_env.sh     # GPU / P2P / nsys / dataset 확인
   nsys --version                # 로컬 Nsight 버전 맞추기용 기록
   ```
3. **스모크 테스트** (nsys 없이 빠르게, 각 모드 2 GPU):
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 python train_cifar.py --num_gpu=2 --data=dataset/cifar10 --ckpt=runs/smoke --mode=ddp
   CUDA_VISIBLE_DEVICES=0,1 python train_cifar.py --num_gpu=2 --data=dataset/cifar10_images --ckpt=runs/smoke --mode=ddp_dali
   ```
   - 확인: 에러 없음, `len(train_loader)` = 98 (=ceil(25000/256)), loss 감소, test acc가 DDP와 DALI에서 비슷한 수준
   - nsys GPU metrics 옵션 동작 여부 확인 → P0 최종 확정
4. **본 실행** (각 스크립트가 1 GPU → 2 GPU 순서로 실행):
   ```bash
   bash run_vastai_dp.sh
   bash run_vastai_ddp.sh
   bash run_vastai_ddp_dali.sh
   ```
   - 문제 발생 시 README Troubleshooting: DDP hang → `export NCCL_P2P_DISABLE=1`, shm 에러 → `export NCCL_SHM_DISABLE=1`
5. **수집** (로컬에서):
   ```bash
   scp -P <PORT> -r root@<IP>:/workspace/CSED490F_DDP_DALI_Training/nsight_logs ./nsight_logs
   scp -P <PORT> -r root@<IP>:/workspace/CSED490F_DDP_DALI_Training/logs ./logs
   ```
   - 인스턴스에서 미리 `nsys stats`로 요약 CSV를 뽑아 같이 받아두면 안전 (아래 Phase D 명령)
6. **destroy** — 다운로드 파일이 열리는지 확인한 뒤 즉시

## Phase D. 분석 & 보고서 계획

### 비교 대상 (총 9개 로그)

| 출처 | 모드 | GPU 수 |
|---|---|---|
| TA 제공 | DP | 1, 2, 4 |
| 내 실행 | DP | 1, 2 |
| 내 실행 | DDP | 1, 2 |
| 내 실행 | DDP+DALI | 1, 2 |

### 추출할 지표

| 지표 | 얻는 방법 |
|---|---|
| Epoch 시간, batch당 시간 | NVTX `Epoch 0`, `Batch i` 구간 (`nsys stats --report nvtx_sum`) |
| 데이터 로딩 시간 | batch 사이 간격 / `Data` 시간 (stdout 로그의 `Data x.xxx`) |
| forward/backward/upload 시간 | NVTX 하위 구간 |
| GPU busy 비율 (≈ GPU util) | GPU metrics 행 (있으면) 또는 `cuda_gpu_trace`의 kernel 시간 합 ÷ epoch 시간 |
| 통신 비중 | NCCL kernel(`ncclKernel_AllReduce…`) 시간 (DDP), DP의 `scatter/gather/broadcast` memcpy P2P 시간 |
| H2D memcpy | `cuda_gpu_mem_time_sum`, `cuda_gpu_mem_size_sum` (DALI는 H2D 이미지 복사가 줄어야 함) |
| 메모리 사용량 | `--cuda-memory-usage` 타임라인, GPU별 peak (DP는 GPU0에 몰림) |

```bash
nsys stats --report nvtx_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
    --format csv --output nsight_logs/stats/<name> <file>.nsys-rep
```

### 보고서 구성 (≤ 4 pages, 11pt)

1. **Implementation** (~1.2p): P0–P7 각 핵심 코드와 설계 결정 (nsys 옵션, per-GPU batch 분할, DistributedSampler, DALI 파이프라인 ↔ torchvision 매핑표, 차이점)
2. **Profiling results** (~1.5p): 표 1개 (모드×GPU수 → epoch time, GPU busy %, 통신 시간, peak mem) + Nsight 타임라인 스크린샷 2–3장 (DP 2GPU / DDP 2GPU / DALI 2GPU NVTX+CUDA 행)
3. **Analysis** (~1.2p): "how and why"
   - **DP**: 단일 프로세스 + GIL, 매 iteration마다 GPU0에서 scatter input / replicate model / gather output → GPU0 메모리·연산 편중, GPU 수↑ → 통신/동기화 overhead↑ → 느려짐(TA 경고 사항). GPU util이 낮고 불균형한 이유를 타임라인 공백으로 설명
   - **DDP**: 프로세스당 GPU, 초기 1회 broadcast + backward 중 bucket 단위 all-reduce overlap → GPU별 부하·메모리 균등, util↑. 단, 데이터 로딩(`num_workers=0`, CPU augmentation)이 병목으로 남아 kernel 사이 공백 존재
   - **DDP+DALI**: augmentation을 GPU로, prefetch로 비동기 → batch 사이 공백 감소, util↑. 대신 DALI 버퍼로 GPU 메모리 사용량 증가 (trade-off). PNG decode는 CPU라는 한계
   - TA 로그(1/2/4 GPU DP)를 해석할 때 통신 overhead와 연산 자체를 분리해서 설명
4. **Conclusion** (~0.1p)

### 제출물

```
lab3_DDP_DALI_team{N}.zip
├── handler/  (DALI, DDP, DP 전체)
├── launch.sh (scripts/launch.sh)
└── lab3_DDP_DALI_team{N}.pdf
```
- 팀 번호 확인 필요

---

## 일정 (안)

| 날짜 | 작업 |
|---|---|
| 9/28 (월) | spec/plan 작성 ✅, Phase A 구현 시작 |
| 9/29 (화) | Phase A 완료, Phase B 검증, commit & push |
| 9/30 (수) | Phase C: Vast.ai 실행 → 로그 수집 → destroy |
| 10/1–10/2 | Phase D: TA 로그 + 내 로그 분석, 그림/표 |
| 10/3 (토) | 보고서 작성 |
| 10/4 (일) | 검토, zip 생성, 제출 (여유 버퍼) |

## 결정 필요 / 열린 이슈

- [x] DDP/DALI의 train batch → per-GPU `512 / world_size` (global batch 512 유지). 보고서에 근거 명시
- [x] DALI seed를 shard마다 다르게 할지 → `12345 + shard_id` 로 결정 (rank별 augmentation 난수열 분리)
- [x] nsys GPU metrics 옵션 → **사용 불가** (`ERR_NVGPUCTRPERM`, 컨테이너 권한 부족). 대신 `nvidia-smi` 0.2초 간격 로그(`logs/nvidia_smi.csv`)로 GPU/Mem util 수집
- [ ] 팀 번호

## 작업 로그

- **2026-09-28**: PDF 명세를 `report/spec.md`로 변환. 코드베이스 분석 후 미구현 지점(P0–P7) 확인 및 본 계획 작성.
- **2026-09-28**: Phase A 구현 완료 (P0–P7).
  - P0: `nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas --cuda-memory-usage=true --sample=none --cpuctxsw=none`. GPU metrics 등 추가 옵션은 `NSYS_EXTRA_ARGS` 환경변수로 넘길 수 있게 함 (예: `NSYS_EXTRA_ARGS="--gpu-metrics-devices=all" bash run_vastai_ddp.sh`)
  - P2: NCCL 바인딩을 위해 `set_device`를 `init_process_group`보다 먼저 호출
  - P5/P7: 모든 로더에서 batch를 `// world_size`로 나누고, `world_size`로 나누어떨어지는지 assert
  - P6: pad = `fn.paste(ratio=1.25)`, flip = `coin_flip` → CMN `mirror`, crop/normalize = CMN (`CHW`, float), cutout = 제공된 `fn_dali_cutout`
  - P7: `LastBatchPolicy.PARTIAL` + `auto_reset=True`. `shuffle` 인자는 쓰지 않음 (train은 항상 shuffle, test는 하지 않음)
  - Phase C에서 확인할 것: `fn.paste`/CMN 출력 shape `(B,3,32,32)`, `len(train_loader)` (2 GPU일 때 98), DDP와 DALI의 loss/acc가 비슷한지, nsys가 spawn된 자식 프로세스를 추적하는지
- **2026-09-29**: Phase C 시작. Vast.ai 인스턴스 준비 확인, `~/.no_auto_tmux` 설정, `/workspace`에 팀 fork clone (`9a05c6d`), `init.sh` 실행.
  - 로컬 Nsight Systems는 인스턴스의 nsys **2025.5.1 이상**이어야 `.nsys-rep`를 열 수 있음
- **2026-09-29**: 스모크 테스트 결과 (2 GPU, 1 epoch, nsys 없음)

  | | DDP (torchvision) | DDP + DALI |
  |---|---|---|
  | iteration 수 | 98 | 98 |
  | train loss / Prec@1 | 1.7676 / 36.85% | 1.7675 / 36.92% |
  | test Prec@1 | 48.46% | 47.60% |
  | batch당 시간 (avg) | 0.181 s | 0.098 s |
  | 그중 data 대기 (avg) | 0.085 s (≈47%) | 0.000 s |

  - DALI 파이프라인이 torchvision과 같은 학습 결과를 냄 → P6 동작 검증 완료
  - `check_env.sh`: GPU 간 P2P access False → GPU 간 복사가 호스트 메모리를 거침 (DP 통신 overhead 분석에 활용)
  - 본 실행 시작 (tmux `main`: dp → ddp → ddp_dali 순서, tmux `smi`: nvidia-smi 로깅)
- **2026-09-29**: 본 실행 완료 (nsys 포함, 1 epoch). 로그 속 값 (Epoch 0의 50번째 iteration 시점 평균)

  | 모드 | GPU | batch당 시간 | data 대기 | train Prec@1 | test Prec@1 |
  |---|---|---|---|---|---|
  | DP | 1 | 0.288 s | 0.171 s | 38.78% | 55.22% |
  | DP | 2 | 0.353 s | 0.171 s | 36.01% | 49.22% |
  | DDP | 1 | 0.287 s | 0.168 s | 37.48% | 48.80% |
  | DDP | 2 | 0.192 s | 0.083 s | 36.78% | 50.48% |
  | DDP+DALI | 1 | 0.123 s | 0.001 s | 34.86% | 46.62% |
  | DDP+DALI | 2 | 0.100 s | 0.001 s | 36.44% | 47.73% |

  - DP는 GPU 2개일 때 오히려 느려짐 (TA가 경고한 통신 overhead와 같은 현상)
  - DP의 loss 값은 `DP/train.py`가 loss를 batch 크기로 한 번 더 나눠 기록해서 작게 보임 (DDP와 단위가 다름)
  - nsys 켜고 실행하면 스모크 테스트보다 느림 (DDP 2 GPU 0.181 → 0.192 s)
  - `nsys stats` CSV (nvtx_sum, cuda_gpu_kern_sum, cuda_gpu_mem_time_sum, cuda_gpu_mem_size_sum, cuda_api_sum) → `nsight_logs/stats/`
- **2026-09-29**: Vast.ai 인스턴스 destroy 확인 (접속 거부). TA 로그 9개 다운로드 완료. 로컬에서 TA 로그를 CSV로 바꾸는 작업은 GUI export로 남겨 둠 (To-do)
