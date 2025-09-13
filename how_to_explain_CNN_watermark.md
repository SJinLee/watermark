# 1. Introduction
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
