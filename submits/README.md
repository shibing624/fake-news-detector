## 1. classic model(baseline)
lr_submit.csv
线下（本地）验证集结果：valid acc: 0.9625779625779626
线上测试集结果：你的提交分数为      0.813572270610746.

xgboost_submit.csv
线下（本地）验证集结果：valid acc: 0.9493243243243243
线上测试集结果：你的提交分数为      0.741375582807775.

catboost_submit.csv
线下（本地）验证集结果：valid acc: 0.9612785862785863
线上测试集结果：你的提交分数为      0.78302590396966.

## 2. classic model(trick)
更新预处理（加去重，去超长句）后：
lr_submit.csv
线下（本地）验证集结果 valid acc: 0.9621258399511301

xgboost_submit.csv
线下（本地）验证集结果 valid acc: 0.9202810018326206

catboost_submit.csv
线下（本地）验证集结果 valid acc: 0.9715943799633476

## 3. deep model
textcnn_submit.csv
线下（本地）验证集结果 valid acc: 0.9801466096518021
耗时：1小时4轮

rnn_submit.csv
线下（本地）验证集结果 valid acc: 0.9737324373854612
耗时：2小时10轮，3轮后无改善

dpcnn_submit.csv
线下（本地）验证集结果 valid acc: 0.9807574832009774
耗时：3小时10轮

bert_submit.csv
线下（本地）验证集结果
耗时：9小时2轮

## 4. stack model
stack_submit.csv
线下（本地）验证集结果