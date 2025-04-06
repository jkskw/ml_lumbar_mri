import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.models.resnet3d import ResidualBlock3D

class ResNetEncoder3D(nn.Module):
    """
    Enkoder oparty na architekturze ResNet3D, który wyciąga mapy cech
    z wejściowego wolumenu. Składa się z początkowej warstwy konwolucyjnej,
    kilku bloków resztkowych i max-poolingu.
    """
    def __init__(self, input_channels=1, dropout_prob=0.3, 
                 block_type=ResidualBlock3D, block_channels=(32, 64, 128), 
                 num_blocks=(3, 4, 6)):
        super(ResNetEncoder3D, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, block_channels[0], 
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(block_channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block_type, block_channels[0], block_channels[0],
                                        num_blocks[0], stride=1, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(block_type, block_channels[0], block_channels[1],
                                        num_blocks[1], stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(block_type, block_channels[1], block_channels[2],
                                        num_blocks[2], stride=2, dropout_prob=dropout_prob)
        
    def _make_layer(self, block_type, in_channels, out_channels, blocks, stride=1, dropout_prob=0.0):
        layers = []
        layers.append(block_type(in_channels, out_channels, stride=stride, dropout_prob=dropout_prob))
        for _ in range(1, blocks):
            layers.append(block_type(out_channels, out_channels, stride=1, dropout_prob=dropout_prob))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x: [B, C, D, H, W]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class AdvancedSpinal3DNetResNetEncoder(nn.Module):
    """
    Proponowana architektura wykorzystująca ResNet jako enkoder, a następnie
    agregację LSTM z mechanizmem uwagi. Struktura:
      - Enkoder ResNetEncoder3D wyciąga mapy cech.
      - Adaptacyjny pooling ustala stały rozmiar (4×12×12).
      - Każdy slice (głębokość) jest spłaszczany i przekazywany do LSTM.
      - Mechanizm uwagi agreguje wyniki LSTM, a wynik trafia do klasyfikatora.
    """
    def __init__(self, num_classes=3, input_channels=1, dropout_prob=0.3):
        super(AdvancedSpinal3DNetResNetEncoder, self).__init__()
        self.encoder = ResNetEncoder3D(input_channels=input_channels, dropout_prob=dropout_prob,
                                       block_type=ResidualBlock3D, block_channels=(32, 64, 128), 
                                       num_blocks=(2, 2, 2))
        # Ustalamy stały rozmiar: głębokość=4, wysokość=12, szerokość=12.
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 12, 12))
        self.dropout3d = nn.Dropout3d(dropout_prob)
        
        # LSTM – wejście: 128*12*12 = 18432 cech, ukryta warstwa 256, dwukierunkowe (512 wyjść)
        self.lstm = nn.LSTM(128 * 12 * 12, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout_lstm = nn.Dropout(dropout_prob)
        
        # Mechanizm uwagi
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Warstwa klasyfikacyjna
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # x: [B, C, D, H, W]
        x = self.encoder(x)              # np. [B, 128, D_enc, H_enc, W_enc]
        x = self.adaptive_pool(x)          # [B, 128, 4, 12, 12]
        x = self.dropout3d(x)
        b, c, d, h, w = x.shape
        # Spłaszczamy wymiary przestrzenne dla każdego slice'a: [B, d, c*h*w]
        x = x.view(b, d, -1)               # [B, 4, 128*12*12]
        lstm_out, _ = self.lstm(x)         # [B, 4, 512]
        lstm_out = self.dropout_lstm(lstm_out)
        attn_scores = self.attention(lstm_out).squeeze(-1)  # [B, 4]
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # [B, 4, 1]
        context = torch.sum(lstm_out * attn_weights, dim=1) # [B, 512]
        out = self.fc(context)
        return out
