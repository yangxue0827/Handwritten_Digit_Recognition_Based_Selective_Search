# Handwritten_Digit_Recognition_with_Selective_Search

# 环境：
Windows10 + tensorflow1.2 + python3.5 + cv2
# 程序：
   example/demo.py---对手写数字图片的分割，并将每个数字做成28*28的黑底白字图片，保存在本地image_data.npy   
   example/mnist_model.py---对手写体mnist数据集进行训练，训练好后读取数据进行识别    
   example/camera.py---是调用计算机摄像头获取图片用的，按q退出拍照    
   selectivesearch/selectivesearch.py---是选择性搜索的源代码
# 注意：
手写数字的图片尽量不要太大（太大会显得数字写的太细，调大数字粗细度），每个数字大小不要差太多，可以在画板上写的一个数字长宽在50像素左右效果不错，其他的没有测试过。
