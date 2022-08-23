其实感觉也没啥好写的，代码相关的基本都在注释里了。哦对了，把代码出处链接放一下，免得自己以后也忘了从哪抄的w

【真-极简爬坡式强化学习入门(代码现编，PyTorch版）】 https://www.bilibili.com/video/BV1Gq4y1v7Bs?share_source=copy_web&vd_source=a75ddd5333dd0777c23d1abe1cbb546f

视频底下还有视频参考的教材和源代码，我就懒得贴那么多了。

2022/08/22

现在的情况是要实现PER，但我发现样本采集出来之后它们的优先级不好更新，然后看别人实现是用的SumTree，所以打算试试

PER写是写出来了，但是跑的巨慢，而且性能并不太理想，可能是因为没有按照优先级顺序去删回放缓存里的内容。

几张结果图

doubleQ，episode 2000（buffer size 5000）

![](DQN-doubleQ/doubleQ2000.png)

doubleQ，episode 3000（buffer size 5000）

![](DQN-doubleQ/doubleQ3000.png)

PER，buffer size 5000（删除最老的）

![](DQN-PER/PERresult.png)

这个效果不怎么好而且巨慢

PER，buffer size 1000（删除最不优先的）

![](DQN-PER/PERresult2.png)

这个也很慢，比5000的好不到哪去

PER效果差的原因可能是因为没加权重

然而加了权重之后仍然巨慢，效果还没出，但看起来也是在700多的时候开始升

PER加了权重，buffer size 1000

![](DQN-PER/PERresult3.png)

还是不明白为什么会有几个跌回0的区间

PER加权重，buffer size 1000，start size 100，这次测试结果有500满奖励

![](DQN-PER/PERresult4.png)