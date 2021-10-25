## 关于密度图的生成

![image-20211025132218333](/home/chenpengyu/.config/Typora/typora-user-images/image-20211025132218333.png)

这个很不合理，会造成生成密度图的sum值与原先的annotation之间存在误差。我认为应该在生成密度图之前完成密度图尺度的变化，如下面的实例所示。

![image-20211025132247041](/home/chenpengyu/.config/Typora/typora-user-images/image-20211025132247041.png)

其中：annotation表示标签点的(x,y)坐标，是亚像素值点

这个是很合理的，是根据放缩的尺度因子，将原先的annotation点映射到新图中的亚像素坐标上。

后面根据亚像素值，合并整点坐标附近的值，记录每个整点坐标对应值的大小。

按照NWPU-Crowd-simple的方法，保存的dot标注文件是作为灰度图保存成png的格式，也可以选择保存成numpy的那种格式，然后在训练的过程中才生成密度图



## 关于多evaluation metrics

在save model时应该选择，针对每种metric都保存一个最佳的模型

