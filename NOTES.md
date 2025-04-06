# Dataset distribution
spinal_canal_stenosis_l1_l2
- Normal/Mild:    1886
- Moderate :        67
- Severe:            21

spinal_canal_stenosis_l2_l3
- Normal/Mild:    1770
- Moderate :       151
- Severe:            53

spinal_canal_stenosis_l3_l4
- Normal/Mild:    1622
- Moderate :       230
- Severe:           122

spinal_canal_stenosis_l4_l5
- Normal/Mild:    1482
- Severe:           255
- Moderate :       237

spinal_canal_stenosis_l5_s1
- Normal/Mild:    1904
- Moderate :        51
- Severe:            19

left_neural_foraminal_narrowing_l1_l2
- Normal/Mild:    1908
- Moderate :        63
- Severe:             2

left_neural_foraminal_narrowing_l2_l3
- Normal/Mild:    1791
- Moderate :       171
- Severe:            11

left_neural_foraminal_narrowing_l3_l4
- Normal/Mild:    1522
- Moderate :       411
- Severe:            40

left_neural_foraminal_narrowing_l4_l5
- Normal/Mild:    1204
- Moderate :       629
- Severe:           140

left_neural_foraminal_narrowing_l5_s1
- Normal/Mild:    1247
- Moderate :       520
- Severe:           206

right_neural_foraminal_narrowing_l1_l2
- Normal/Mild:    1891
- Moderate :        63
- Severe:            13

right_neural_foraminal_narrowing_l2_l3
- Normal/Mild:    1793
- Moderate :       168
- Severe:             6

right_neural_foraminal_narrowing_l3_l4
- Normal/Mild:    1512
- Moderate :       414
- Severe:            41

right_neural_foraminal_narrowing_l4_l5
- Normal/Mild:    1208
- Moderate :       629
- Severe:           130

right_neural_foraminal_narrowing_l5_s1
- Normal/Mild:    1281
- Moderate :       496
- Severe:           190

left_subarticular_stenosis_l1_l2
- Normal/Mild:    1690
- Moderate :        93
- Severe:            28

left_subarticular_stenosis_l2_l3
- Normal/Mild:    1555
- Moderate :       255
- Severe:            83

left_subarticular_stenosis_l3_l4
- Normal/Mild:    1324
- Moderate :       454
- Severe:           194

left_subarticular_stenosis_l4_l5
- Normal/Mild:    887
- Moderate :      624
- Severe:          461

left_subarticular_stenosis_l5_s1
- Normal/Mild:    1408
- Moderate :       409
- Severe:           147

right_subarticular_stenosis_l1_l2
- Normal/Mild:    1680
- Moderate :       110
- Severe:            24

right_subarticular_stenosis_l2_l3
- Normal/Mild:    1577
- Moderate :       243
- Severe:            73

right_subarticular_stenosis_l3_l4
- Normal/Mild:    1322
- Moderate :       454
- Severe:           197

right_subarticular_stenosis_l4_l5
- Normal/Mild:    891
- Moderate :      622
- Severe:          460

right_subarticular_stenosis_l5_s1
- Normal/Mild:    1399
- Moderate :       396
- Severe:           173


# Single Binary
### SCS
**Model**:
- ./models/ResNet3D_34_single_binary_target_window_128x128_5D_B2A2_4_100_0.0001_0.3_20250320_193734/best_val_loss_model.pth

- ./models/AdvancedSpinal3DNetImproved_single_binary_target_window_128x128_3D_B1A1_16_100_0.001_0.5_20250320_100014/best_val_loss_model.pth 

- models/AdvancedSpinal3DNetImproved_single_binary_target_window_128x128_3D_B1A1_16_200_0.0001_0.3_20250319_122303

- models/AdvancedSpinal3DNetImproved_single_binary_target_window_128x128_3D_B1A1_16_200_0.0001_0.3_20250319_112138

-  models/advanced_single_improved_single_binary_target_window_128x128_3D_B1A1_16_30_0.001_0.5_20250310_172516

- models/advanced_single_improved_single_binary_target_window_128x128_3D_B1A1_16_30_0.001_0.3_20250308_213637

- models/advanced_single_improved_single_binary_target_window_96x96_3D_B1A1_16_30_0.001_0.3_20250308_224733

- models/advanced_single_improved_single_binary_target_window_128x128_3D_B1A1_16_30_0.001_0.5_20250308_230149

### RNFN
**Model**:
- models/advanced_single_improved_single_binary_target_window_128x128_5D_B2A2_16_200_0.001_0.5_20250310_142702

- models/advanced_single_improved_single_binary_target_window_128x128_5D_B2A2_16_30_0.001_0.5_20250309_171258

### LNFN
**Model**:
- models/advanced_single_improved_single_binary_target_window_128x128_5D_B2A2_16_200_0.001_0.5_20250310_135914

- models/advanced_single_improved_single_binary_target_window_128x128_5D_B2A2_16_30_0.001_0.5_20250309_173018

# Single Multiclass
## SCS
**Model**:
- wszystko ./models/cv_ResNeXtEncoderBiLSTMClassifier_single_multiclass_target_window_128x128_5D_B2A2_16_35_0.0001_0.5

- 87% L3L4 ./models/ResNeXtEncoderBiLSTMClassifier_single_multiclass_target_window_128x128_5D_B2A2_16_100_0.001_0.5_20250321_180254/best_bal_acc_model.pth

- 79% ./models/AdvancedSpinal3DNetResNetEncoder_single_multiclass_target_window_128x128_5D_B2A2_64_100_0.0001_0.5_20250321_172032/best_val_loss_model.pth

- crosval models/cv_AdvancedSpinal3DNetResNetEncoder_single_multiclass_target_window_128x128_5D_B2A2_64_30_0.0001_0.5/training.log 


- 75% ./models/AdvancedSpinal3DNetResNetEncoder_single_multiclass_target_window_128x128_5D_B2A2_16_100_0.0001_0.5_20250321_163505/best_bal_acc_model.pth

- 74% ./models/AdvancedSpinal3DNetResNetEncoder_single_multiclass_target_window_128x128_5D_B2A2_32_100_0.0001_0.5_20250321_164456/best_bal_acc_model.pth

- ./models/AdvancedSpinal3DNetResNetEncoder_single_multiclass_target_window_128x128_5D_B2A2_32_100_0.0001_0.5_20250321_165635/best_bal_acc_model.pth


- ./models/AdvancedSpinal3DNetResNetEncoder_single_multiclass_target_window_128x128_5D_B2A2_64_100_0.0001_0.5_20250321_134733/best_val_loss_model.pth

- ./models/AdvancedSpinal3DNetResNetEncoder_single_multiclass_target_window_128x128_5D_B2A2_16_100_0.0001_0.5_20250321_145112

- models/ConvNeXtSmallLSTM_single_multiclass_target_window_128x128_5D_B2A2_4_100_0.0001_0.3_20250319_204809/best_val_loss_model.pth 

- models/advanced_single_improved_single_multiclass_target_window_128x128_3D_B1A1_16_200_0.001_0.5_20250313_000651

- models/advanced_single_improved_single_multiclass_target_window_128x128_3D_B1A1_16_200_0.001_0.5_20250310_174007

### RNFN
**Model**:
- ./models/cv_AdvancedSpinal3DNetResNetEncoder_single_multiclass_target_window_128x128_5D_B2A2_64_30_0.0001_0.5

- ./models/cv_AdvancedSpinal3DNetResNetEncoder_single_multiclass_target_window_128x128_5D_B2A2_64_30_0.0001_0.5

- models/advanced_single_improved_single_multiclass_target_window_128x128_3D_B1A1_16_200_0.001_0.5_20250310_190532

- models/advanced_single_improved_single_multiclass_target_window_128x128_3D_B1A1_16_200_0.001_0.5_20250310_191526 - **use_class_weights: true**

### LNFN
**Model**:
- models/advanced_single_improved_single_multiclass_target_window_128x128_3D_B1A1_16_200_0.001_0.5_20250310_192311

- models/advanced_single_improved_single_multiclass_target_window_128x128_3D_B1A1_16_200_0.001_0.5_20250310_193221 - **use_class_weights: true**


# Multi Multiclass
**Model**:
- models/resnet3d_multi_multiclass_target_window_128x128_5D_B2A2_16_200_0.001_0.5_20250309_191331

- models/resnet3d_multi_multiclass_target_window_128x128_5D_B2A2_32_30_0.001_0.3_20250308_191714

- models/resnet3d_multi_multiclass_full_series_128x128_19D_16_200_0.001_0.3_20250309_182709

- models/resnet3d_multi_multiclass_target_window_128x128_5D_B2A2_16_200_0.001_0.5_20250309_200317 - **30% test**

models/resnet3d_50_multi_multiclass_full_series_128x128_15D_16_200_0.001_0.5_20250312_213119

ResNets
- models/resnet3d_18_multi_multiclass_target_window_128x128_3D_B1A1_32_200_0.001_0.5_20250311_154831

**Badania wpływu głębkości Tensora**

**1:** models/resnet3d_multi_multiclass_full_series_128x128_1D_16_200_0.001_0.5_20250309_203836

**3:** models/resnet3d_multi_multiclass_target_window_128x128_3D_B1A1_16_200_0.001_0.5_20250309_210156

**128x128 5:** models/resnet3d_multi_multiclass_target_window_128x128_5D_B2A2_16_200_0.001_0.5_20250309_191331

**96x96 5:** models/resnet3d_multi_multiclass_target_window_96x96_5D_B2A2_16_200_0.001_0.5_20250309_213545

**15:** models/resnet3d_multi_multiclass_full_series_128x128_15D_16_200_0.001_0.5_20250309_220008

**17:** models/resnet3d_multi_multiclass_full_series_128x128_17D_16_200_0.001_0.5_20250309_232759

**19:** models/rresnet3d_multi_multiclass_full_series_128x128_19D_16_200_0.001_0.5_20250310_003204

**25:** models/resnet3d_multi_multiclass_full_series_128x128_25D_16_200_0.001_0.5_20250310_013908

# Multi Single
**Model**:
- models/resnet3d_multi_binary_target_window_128x128_3D_B1A1_16_200_0.001_0.5_20250310_094513

- models/resnet3d_multi_binary_target_window_128x128_5D_B2A2_16_200_0.001_0.5_20250310_101319

- models/advanced_multi_multi_binary_target_window_128x128_5D_B2A2_16_200_0.001_0.5_20250310_105210

- models/resnet3d_multi_multiclass_target_window_128x128_5D_B2A2_16_200_0.001_0.5_20250309_200317 - **30% test**