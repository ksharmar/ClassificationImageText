Waseem 

** Labels: 'racism':0, 'sexism':1, 'both':2, 'neither':3

** Noise Rate (total) = 13.88 

(1) but more noise in smaller classes
(2) lots of false positives in racism and sexism (gt=neither, but put into these).

** Expert dist = 6909 total (Imbalance)
98 (1.411)
911 (13.189)
50 (0.7237)
5850 (84.675)

Errors in each class (how many given labels marked 0-3 type are incorrect)
146 / 189 (77%)
576 / 1312 (43%)
10 / 19 (52%)
227 / 5389 (4%)

      r    s     b   n
r [[ 43    4    0   51]
s [  15  736    3  157]
b [   8   14    9   19]
n [ 123  558    7 5162]]

** Noise Transition Matrix (expert -> majority what % are moved to wrong class)
[[43.88  4.08  0.   52.04]
 [ 1.65 80.79  0.33 17.23]
 [16.   28.   18.   38.  ]
 [ 2.1   9.54  0.12 88.24]]


**  
Noisy Train 5527 (=verified val) for detection task
Clean Test 1382 (classification task)

** Classification (baseline noisy train | eval on gt_test): 6e

Test Accuracy: 0.8224137931034482
('              precision    recall  f1-score   support\n'
 '\n'
 '           0       0.26      0.60      0.36        20\n'
 '           1       0.46      0.91      0.62       182\n'
 '           2       0.00      0.00      0.00        10\n'
 '           3       0.98      0.82      0.89      1169\n'
 '\n'
 '    accuracy                           0.82      1381\n'
 '   macro avg       0.43      0.58      0.47      1381\n'
 'weighted avg       0.90      0.82      0.84      1381\n')
matthews corr coeff: 0.5596866977213429
finished program.

** Classification (upper bound gt_train | eval on gt_test): 6e


Test Accuracy: 0.8748563218390805
('              precision    recall  f1-score   support\n'
 '\n'
 '           0       0.50      0.65      0.57        20\n'
 '           1       0.56      0.80      0.66       182\n'
 '           2       0.50      0.20      0.29        10\n'
 '           3       0.96      0.90      0.93      1169\n'
 '\n'
 '    accuracy                           0.88      1381\n'
 '   macro avg       0.63      0.64      0.61      1381\n'
 'weighted avg       0.90      0.88      0.88      1381\n')
matthews corr coeff: 0.6059304968022828



CLEANNET

INFO:tensorflow:acc = 0.8867722602739726 err = 0.1132277397260274
INFO:tensorflow:recall = 0.4122137404580153
INFO:tensorflow:precision = 0.6521739130434783
INFO:tensorflow:f1_noise = 0.5051449953227315
INFO:tensorflow:avg acc (cls) = 0.6795090397305888 avg error (cls) = 0.32049096026941115
f1_metrics (clean) 0.9360725075528702
f1_metrics (noise) 0.5051449953227315
f1_metrics (accuracy/micro) 0.8867722602739726
f1_metrics (macro/unweighted mean) 0.7206087514378008
f1_metrics (weighted mean of f1) 0.875657798539441


AVG BASELINE

INFO:tensorflow:Start avg baseline validate once...
INFO:tensorflow:Get data batcher...
>> Predict for 4672 batches.
INFO:tensorflow:acc = 0.8035102739726028 err = 0.19648972602739723
INFO:tensorflow:recall = 0.3938931297709924
INFO:tensorflow:precision = 0.33119383825417203
INFO:tensorflow:f1_noise = 0.35983263598326365
INFO:tensorflow:avg acc (cls) = 0.6156793401189902 avg error (cls) = 0.38432065988100983
f1_metrics (clean) 0.8839443742098609
f1_metrics (noise) 0.35983263598326365
f1_metrics (accuracy/micro) 0.8035102739726028
f1_metrics (macro/unweighted mean) 0.6218885050965623
f1_metrics (weighted mean of f1) 0.8104655239233838


OURS








