# pytorch - cifar-10/100

		* cifar-10  分别使用 VGG、googLeNet、Alexnet 进行训练
		* cifar-100 使用 googLeNet、resnet、se+resnet

* visdom可视化
   >    `python -m visdom.server`
   >    You can navigate to http://localhost:8097
<br>    由于时间限制更换`resnet152`只进行了一次10个epoch的完全训练，最优结果58%，出现过拟合的情况，参数待调整
<br>	尝试通过更改BATCHSIZE来减轻过拟合的情况，发现现有配置`GTX970M`所能计算的最大size为10
![可视化结果](https://raw.githubusercontent.com/NIJUNPEI/pytorch_cifar/master/cifar-100/senet%2Bresnet/visdom.jpg)
<br>	更换`resnet101`BATCHSIZE更改为20,训练15个epoch，最优结果59.2%
<br>	在第7次训练之后出现过拟合现象
>### 七次训练loss结果
![loss](https://raw.githubusercontent.com/NIJUNPEI/pytorch_cifar/master/cifar-100/senet%2Bresnet/se%2Bresnet101(visdom-loss).jpg)
>### 七次训练acc 结果
![acc](https://raw.githubusercontent.com/NIJUNPEI/pytorch_cifar/master/cifar-100/senet%2Bresnet/se%2Brenest101(visdom-acc).jpg)
