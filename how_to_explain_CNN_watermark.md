# 1. Introduction
＃＃ １。１ 인공신경망의구조
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

- 학습데이터
   - 호스트 영상의 학습 데이터로 그레이(gray) 스케일 영상이10,000장으로 구성된 BOSS 데이터 셋을 128×128 해상도로 스케일링(scaling)하여 사용
      - [https://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip](https://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip)
   - 평가(test) 데이터로그레이(gray) 스케일 영상이 49장으로 구성된 표준 시험데이터셋을 128×128 해상도로 스케일링하여 사용
      - [http://decsai.ugr.es/cvg/CG/base.htm](http://decsai.ugr.es/cvg/CG/base.htm)
   － 빠른 학습을 위해 학습데이터 중에서 ９개、 평가 데이터 중 ３개의 이미지를 사용함
