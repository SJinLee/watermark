# 1. Introduction
# 1.1 워터마크란?
- 워터마크의 개념:
- 워터마크 삽입방법:
  1. 텍스트 워터마크

  - 저작권 정보, 회사명, URL 등을 텍스트로 삽입
  - 투명도 조절로 가독성과 보호 균형

  2. 이미지 워터마크

  - 로고, 서명, 스탬프 등을 이미지로 오버레이
  - PNG 투명 배경 활용

  3. 위치별 분류

  - 코너 워터마크: 모서리에 작게 배치
  - 센터 워터마크: 중앙에 크게 배치
  - 반복 워터마크: 전체에 패턴으로 반복

  4. 구현 방법

  - GUI 도구: Photoshop, GIMP, Canva
  - 온라인 서비스: 웹 기반 워터마킹 도구
  - 프로그래밍: Python (PIL/Pillow), JavaScript (Canvas), ImageMagick
  - 배치 처리: 여러 이미지 일괄 처리

  5. 고급 기법

  - 보이지 않는 워터마크: 스테가노그래피 활용
  - 디지털 서명: 메타데이터에 저작권 정보 삽입
  - 블렌딩 모드: 다양한 합성 효과

## 1.1 인공신경망의구조
- [워터마크 및 해상도 적응적인 영상 워터마킹을 위한 딥 러닝 프레임워크](https://scienceon.kisti.re.kr/commons/util/originalView.do?cn=JAKO202011263332864&dbt=JAKO&koi=KISTI1.1003%2FJNL.JAKO202011263332864)
   의 워터마크 인공신경망을 수정하여 사용하였음
- Host이미지 전처리 네트워크, 워터마크 전처리 네트워크, 워터마크 삽입네트워크, 공격 시뮬레이션, WM 추출 네트워크로 인공신경망은 구성되었고 CNN만으로 이루어졌음
- Host이미지는 128x128크기의 gray scale
- 워터마크는 8x8크기의 bit pattern

<img width="250" src="framework.png">

- 각 네트워크들은 ConvBlock과 ConvTransposeBlock으로 구성됨
```
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU() if activation else None
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x
class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, activation=True):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU() if activation else None
    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x
```

- Host 전처리 네트워크:
```
ConvBlock(1, 16, kernel_size=3, stride=2, padding=1, activation=True) # 128x128 -> 64x64
ConvBlock(16, 8, kernel_size=3, stride=1, padding=1, activation=False) # 64x64 -> 64x64
```
- Watermark 전처리 네트워크:
```
ConvTransposeBlock(1, 16, kernel_size=4, stride=2, padding=1, activation=True) # 8x8 -> 16x16
ConvTransposeBlock(16, 32, kernel_size=4, stride=2, padding=1, activation=True) # 16x16 -> 32x32
ConvTransposeBlock(32, 8, kernel_size=4, stride=2, padding=1, activation=False) # 32x32 -> 64x64
```

- 워터마크 삽입 네트워크
```
ConvBlock(16, 32, kernel_size=3, stride=1, padding=1, activation=True) # 64x64 -> 64x64
ConvBlock(32, 16, kernel_size=3, stride=1, padding=1, activation=True) # 64x64 -> 64x64
ConvTransposeBlock(16, 1, kernel_size=4, stride=2, padding=1, activation=False) # 64x64 -> 128x128
```

- 워터마크 추출 네트워크
```
ConvBlock(1, 32, kernel_size=3, stride=2, padding=1, activation=True)   # 128->64
ConvBlock(32, 64, kernel_size=3, stride=2, padding=1, activation=True)  # 64->32
ConvBlock(64, 128, kernel_size=3, stride=2, padding=1, activation=True) # 32->16
ConvBlock(128, 1, kernel_size=3, stride=2, padding=1, activation=False) # 16->8
```

- Loss 함수
   - 삽입 네트워크의 출력과 host 이미지 사이의 loss는 MSE(Mean Squared Error)를 사용하였고 이것을 비가시성 에러라고 함
   - 추출 네트워크의 출력과 워터마크 사이의 loss는 MSE(Mean Squared Error)를 사용하였고 이것을 강인성 에러라고 함
   - 전체 loss는 두 에러의 가중합을 사용함
      - [전체 loss]  = 1.0 * [비가시성 에러] + 0.5 * [강인성 에러]

- 최적화 방법
   - Adam을 사용함

## 1.2 학습데이터
- 호스트 영상의 학습 데이터로 그레이(gray) 스케일 영상이10,000장으로 구성된 BOSS 데이터 셋을 128×128 해상도로 스케일링(scaling)하여 사용
   - [https://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip](https://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip)
- 평가(test) 데이터로그레이(gray) 스케일 영상이 49장으로 구성된 표준 시험데이터셋을 128×128 해상도로 스케일링하여 사용
   - [http://decsai.ugr.es/cvg/CG/base.htm](http://decsai.ugr.es/cvg/CG/base.htm)
- 빠른 학습을 위해 학습데이터 중에서 ９개、 평가 데이터 중 ３개의 이미지를 사용함
- 
## 1.3 질문
 - 학습데이터가 작아도 되는 이유
 - watermark delta의 모양
    -  주기적인가?
    -  워터마크의 값이 0일때와 1일 때의 차이가 있는가?
    -  host image의 밝기에 따라 다른가?
    -  어떤 가중치가 중요한가? 가중치를 바꾸면 extraction이 잘 안되는 경우가 있는가?
    -  어떤 가중치를 바꿀때 embedding 결과가 크게 달라지는가?

