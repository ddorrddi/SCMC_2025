### 새로 다운 받은 경우 build 하는 법

#### 1. repo 내 서브 모듈 다운로드
```
git submodule update --init --recursive
```

#### 2. 의존성 설치하기
```
rosdep install --from-paths . --ignore-src -r -y
```

#### 3. catkin_make or catkin build
# SCMC_2025
