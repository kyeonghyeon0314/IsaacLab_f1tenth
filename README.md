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
- 충돌 페널티: -30%
