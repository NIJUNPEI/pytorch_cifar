# pytorch - cifar-10/100

		* cifar-10  分别使用 VGG、googLeNet、Alexnet 进行训练
		* cifar-100 使用 googLeNet、resnet、se+resnet

* visdom可视化
   >    `python -m visdom.server`
   >    You can navigate to http://localhost:8097
<br>    由于时间限制更换`resnet152`只进行了一次10个epoch的完全训练，结果出现过拟合的情况，参数待调整
![可视化结果](https://raw.githubusercontent.com/NIJUNPEI/pytorch_cifar/master/cifar-100/senet%2Bresnet/newplot.png)
