### 前言
*  总觉得我的Github空空如也的就想传点东西，
   然后翻到这个，是以前的课程作业弄得Tensorflow Demo和去年暑假无聊学Qt合并而成的东西。
   就想着再看一看，结果大半年不见把自己绕晕了，花了点时间重新看懂，
   沿袭课程作业的习惯加了一些初中生英文注释。要是有人看到，别笑。
* Tensorflow Demo 基本就是官网Demo改的，QT也都是些基本用法，
   所以我个人觉得适合新手看一看。
___
### 正式介绍
我在的课题组不做模式识别，做的基本是基于数学模型的故障检测，
所以课程作业时另辟蹊径的想试试基于数据的故障检测，正好在kaggle上看见一个使用
1D CNNs，也就时一维CNNs检测故障的例子，本质就是把数据看成一维的图片进行二分类。然后我就在kaggle上找了一个“2013欧洲信用卡信息”
的数据集，它包含492个违规信息和284807个正常信息，然后选取这492个违规信息和另外4920个正常信息
做数据集。
**使用的是tensorflow1.4，1.3版本貌似也能跑。**
 * 数据预处理请参看 data_load_credit.py 中的load_data函数。
   这里我还测试了tfrecords格式文件的使用，代码的原型参考自这个博客：
   > https://blog.csdn.net/u012759136/article/details/52232266
 * 运行data_load_credit.py中的main函数会自动创建需要的tfrecords.
   main函数之后创建了一个tensorflow的队列，使用两个方法读取tfrecords中的数据。
 * 主程序是VGG_credit.py,可以直接运行，里面仿造tensorflow官网Demo搭了一个简化版VGG16网络，
   具体的细节请参看程序，注解很详细。原来的程序使用matplotlib显示的，
   程序里全部注解掉了，换成了QT界面。
 * 需要PyQt5的包，界面是QT和eric6弄的，matplotlib的QT应用代码参看官方文档。
 * 为了让跑训练的时候，Qt界面不至于死掉(那种鼠标拖不动的感觉你懂的)，我给训练独立分配了一个线程，
   用两个队列交换数据，Qt里的定时器会定时取出然后打印在界面上。
 * 这个模型训练的效果实话说不好，但作为一个Demo，看一看就行了。
