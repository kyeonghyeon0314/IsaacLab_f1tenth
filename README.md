# IsaacLab 도커 사용시 주의사항
./docker/container.py start ros2 실행시 

## 마운트 설정
docker-compose.yaml 파일을 보면 다음과 같이 bind 마운트가 설정되어 있습니다:
- type: bind

  source: ../source

  target: ${DOCKER_ISAACLAB_PATH}/source

- type: bind

  source: ../scripts

  target: ${DOCKER_ISAACLAB_PATH}/scripts

- type: bind

  source: ../docs

  target: ${DOCKER_ISAACLAB_PATH}/docs

- type: bind

  source: ../tools

  target: ${DOCKER_ISAACLAB_PATH}/tools

## 실제 경로 매핑
.env.base에서 DOCKER_ISAACLAB_PATH=/workspace/isaaclab로 정의되어 있으므로:

- 컨테이너: /workspace/isaaclab/source ↔ 로컬: /home/kimkh/IsaacLab/source
- 컨테이너: /workspace/isaaclab/scripts ↔ 로컬: /home/kimkh/IsaacLab/scripts
- 컨테이너: /workspace/isaaclab/docs ↔ 로컬: /home/kimkh/IsaacLab/docs
- 컨테이너: /workspace/isaaclab/tools ↔ 로컬: /home/kimkh/IsaacLab/tools

결론: 컨테이너 내부에서 /workspace/isaaclab 디렉토리의 하위 폴더들(source, scripts, docs, tools)은 현재 로컬의 /home/kimkh/IsaacLab 디렉토리와 실시간으로 동기화됩니다. 즉, 컨테이너에서 파일을 수정하면 로컬에도 반영되고, 로컬에서 수정해도 컨테이너에 즉시 반영됩니다.

# 현재 모델 설계 
## F1TENTH 레이싱 환경 (f1tenth_env.py)
### 관측 공간 (Observation):
- LiDAR 센서: 1080개 ray (270도 스캔, 0.25도 해상도)
- 차량 상태: 속도(speed) + 조향각(steering)
- 총 차원: 1082차원 (1080 + 2)
### 행동 공간 (Action):
- 조향 속도 (steering velocity): [-3.2, 3.2] rad/s
- 가속도 (acceleration): [-9.51, 9.51] m/s²
- 총 차원: 2차원 (연속 행동 공간)
### 물리 시뮬레이션:
- 차량 동역학 모델 (f1tenth_gym 기반)
- VESC 스타일 속도 제어
- 실제 F1TENTH 하드웨어 파라미터 사용
### 보상 함수:
- 진행 보상: 60% (전진 거리 기반)
- 속도 보상: 10% (빠른 주행 장려)
- 저속 페널티: -2.0/step (0-1m/s), -0.5/step (1-2m/s), 패널티 없음 (2m/s 이상)
- 벽 근접 페널티: -1.5/step (25cm 이내), -20/step (10cm 이내)
- 충돌 페널티: 제거 (연속 페널티로 대체)

---

# SAC 3단계 커리큘럼 학습 사용법

## 개요
CNN과 MLP를 동시에 학습하는 것은 비효율적이므로, **3단계로 나누어 점진적으로 학습**하는 커리큘럼 전략을 사용합니다.

### 학습 전략
1. **Stage 1**: 간단한 각도 기반 특징 + MLP 학습 (기초 제어)
2. **Stage 2**: CNN 학습 + MLP 동결 (복잡한 특징 추출)
3. **Stage 3**: CNN + MLP 동시 미세조정 (최종 최적화)

---

## Stage 1: MLP 학습 (각도 기반 특징)

### 설명
- 1080개 LiDAR ray를 128개 각도 구간으로 단순화
- MLP만 학습하여 기본 주행 능력 습득
- 목표: 벽 회피 및 전진 주행

### 실행 명령어
```bash
cd /home/kimkh/IsaacLab

# 학습 시작 (헤드리스 모드)
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-F1tenth-Direct-v0 \
    --algorithm SAC \
    --num_envs 32 \
    --stage 1 \
    --headless

# 시각화 모드로 학습 (학습 과정 확인)
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-F1tenth-Direct-v0 \
    --algorithm SAC \
    --num_envs 8 \
    --stage 1
```

### 학습 결과
- **체크포인트 저장 위치**: `logs/skrl/f1tenth_sac/[날짜_시간]/checkpoints/best_agent.pt`
- **권장 학습 스텝**: 50,000 steps (약 2-3시간)
- **주요 지표**: 랩 완주율, 평균 생존 시간

**중요**: Stage 2를 위해 체크포인트 경로를 저장해두세요!

---

## Stage 2: CNN 학습 (MLP 동결)

### 설명
- Stage 1의 MLP 가중치를 불러와서 **동결** (학습 안 함)
- CNN만 학습하여 LiDAR 공간 패턴 추출
- 목표: 이미 작동하는 제어를 유지하면서 CNN 학습

### 실행 명령어
```bash
# Stage 1 체크포인트 경로를 지정하여 학습 시작
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-F1tenth-Direct-v0 \
    --algorithm SAC \
    --num_envs 32 \
    --stage 2 \
    --mlp_checkpoint logs/skrl/f1tenth_sac/[STAGE1_날짜_시간]/checkpoints/best_agent.pt \
    --headless
```

### 예제
```bash
# 실제 사용 예시
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-F1tenth-Direct-v0 \
    --algorithm SAC \
    --num_envs 32 \
    --stage 2 \
    --mlp_checkpoint logs/skrl/f1tenth_sac/2025-10-27_00-50-01_sac_torch/checkpoints/best_agent.pt \
    --headless
```

### 학습 결과
- **체크포인트 저장 위치**: `logs/skrl/f1tenth_sac/[날짜_시간]/checkpoints/best_agent.pt`
- **권장 학습 스텝**: 100,000 steps (약 4-6시간)
- **주요 지표**: 충돌 감소, 벽 근접 패널티 감소

**중요**: Stage 3를 위해 이 체크포인트 경로도 저장해두세요!

---

## Stage 3: CNN + MLP 미세조정

### 설명
- Stage 1의 MLP + Stage 2의 CNN 가중치를 모두 불러옴
- 둘 다 낮은 학습률로 동시 미세조정
- 목표: End-to-end 최적화, 랩타임 개선

### 실행 명령어
```bash
# Stage 1과 Stage 2 체크포인트를 모두 지정
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-F1tenth-Direct-v0 \
    --algorithm SAC \
    --num_envs 32 \
    --stage 3 \
    --mlp_checkpoint logs/skrl/f1tenth_sac/[STAGE1_날짜_시간]/checkpoints/best_agent.pt \
    --cnn_checkpoint logs/skrl/f1tenth_sac/[STAGE2_날짜_시간]/checkpoints/best_agent.pt \
    --headless
```

### 예제
```bash
# 실제 사용 예시
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-F1tenth-Direct-v0 \
    --algorithm SAC \
    --num_envs 32 \
    --stage 3 \
    --mlp_checkpoint logs/skrl/f1tenth_sac/2025-10-27_00-50-01_sac_torch/checkpoints/best_agent.pt \
    --cnn_checkpoint logs/skrl/f1tenth_sac/2025-10-27_05-30-15_sac_torch/checkpoints/best_agent.pt \
    --headless
```

### 학습 결과
- **최종 모델 저장 위치**: `logs/skrl/f1tenth_sac/[날짜_시간]/checkpoints/best_agent.pt`
- **권장 학습 스텝**: 100,000 steps (약 4-6시간)
- **주요 지표**: 랩타임, 평균 속도, 충돌률

---

## 학습률 설정 (자동)

### Stage 1 (MLP만)
- Policy MLP: 1e-4 (고정)
- Critic MLP: 5e-4 (고정)

### Stage 2 (CNN만)
- Policy CNN: 1e-4 (지수 감쇠, gamma=0.99995)
- Critic CNN: 5e-4 (지수 감쇠, gamma=0.99995)
- MLP: 동결 (학습 안 함)

### Stage 3 (CNN + MLP)
- Policy CNN: 5e-5 (지수 감쇠, gamma=0.99998)
- Policy MLP: 1e-5 (고정)
- Critic CNN: 2.5e-4 (지수 감쇠, gamma=0.99998)
- Critic MLP: 5e-5 (고정)

---

## 네트워크 아키텍처

### CNN (LiDAR Feature Extractor)
```
Input: (batch, 1, 1080)
Conv1D: 1 → 32 channels (kernel=5, stride=2)
Conv1D: 32 → 64 channels (kernel=3, stride=2)
Conv1D: 64 → 128 channels (kernel=3, stride=2)
Flatten + FC: 17280 → 128 features
```

### Policy MLP
```
Input: 128 (LiDAR) + 2 (vehicle state) = 130
Hidden: 130 → 128 → 128 → 64
Output: 2 actions (steering velocity, acceleration)
```

### Critic MLP
```
Input: 128 (LiDAR) + 2 (vehicle state) + 2 (actions) = 132
Hidden: 132 → 256 → 256 → 128
Output: 1 (Q-value)
```

---

## 학습 모니터링

### TensorBoard 사용
```bash
# 학습 중 실시간 모니터링
tensorboard --logdir logs/skrl/f1tenth_sac
```

### 주요 메트릭
- **Stage 1**: `Reward/episode`, `Episode_length`, `Collision_rate`
- **Stage 2**: `Danger_penalty`, `Min_distance_to_wall`, `Collision_rate`
- **Stage 3**: `Lap_time`, `Average_speed`, `Episode_reward`

### 학습률 로그 (자동 출력)
- 1000 스텝마다 현재 학습률이 콘솔에 출력됩니다
- 예: `[LR] Step 5000: Policy(CNN=9.75e-05, MLP=1.00e-05), Critic(CNN=2.44e-04, MLP=5.00e-05)`

---

## 체크포인트 관리

### 자동 저장
- `best_agent.pt`: 최고 성능 모델 (Reward 기준)
- `agent_10000.pt`: 10000 스텝마다 주기적 저장

### 학습 재개
```bash
# 중단된 학습 이어서 하기
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-F1tenth-Direct-v0 \
    --algorithm SAC \
    --num_envs 32 \
    --stage 2 \
    --checkpoint logs/skrl/f1tenth_sac/[날짜_시간]/checkpoints/agent_10000.pt \
    --headless
```

---

## 예상 학습 시간 (단일 GPU)

| Stage | 스텝 수 | 예상 시간 | 목표 |
|-------|--------|-----------|------|
| Stage 1 | 50,000 | 2-3시간 | 랩 완주 능력 |
| Stage 2 | 100,000 | 4-6시간 | 충돌 회피 개선 |
| Stage 3 | 100,000 | 4-6시간 | 랩타임 최적화 |
| **합계** | **250,000** | **10-15시간** | **최종 모델** |

---

## 트러블슈팅

### 문제 1: Stage 2/3에서 체크포인트 로딩 실패
```
해결: 체크포인트 경로가 정확한지 확인
ls logs/skrl/f1tenth_sac/*/checkpoints/
```

### 문제 2: Stage 2에서 성능 저하
```
해결: Stage 1이 충분히 학습되지 않았을 수 있음
Stage 1을 더 오래 학습하거나 MLP 네트워크 크기 확인
```

### 문제 3: GPU 메모리 부족
```
해결: --num_envs를 줄이기 (32 → 16 → 8)
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-F1tenth-Direct-v0 \
    --algorithm SAC \
    --num_envs 16 \
    --stage 1 \
    --headless
```

### 문제 4: 학습이 너무 느림
```
해결: 헤드리스 모드 사용 (--headless)
GPU 사용 확인: nvidia-smi
```
