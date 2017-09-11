

******************************************

# To run a Tensorflow graph and view it in TensorBoard

#Go to this chapter folder

cd chapter_09_folder_path

docker run -it   -p 8888:8888 -p 6006:6006  -v /$(pwd):/notebooks   gcr.io/tensorflow/tensorflow bash

root@......:/notebooks# ipython notebook  --allow-root

#browse :

http://localhost:8888/?token=......

#run sample_graph.ipynb

#Ctrl+C

#/notebooks/output is the output of saved graph

root@......:/notebooks#tensorboard --logdir /notebooks/output
#browse

http://localhost:6006/#graphs

#When you are down Ctrl +C

 rm -rf output/

#follow same procedure with linear_regression.ipynb

********************************
# CIFAR-10 image classifier

#Go to this chapter folder

cd chapter_09_folder_path

docker run -it -p 8888:8888 -p 6006:6006 -v /$(pwd):/notebooks --name tf gcr.io/tensorflow/tensorflow bash

root@......:/notebooks#  pip install tflearn

root@......:/notebooks# ipython notebook  --allow-root





************************************

# Image classification

#Go to this chapter folder

cd chapter_09_folder_path

#Clone tensorflow model from git

cd test_classify

git clone https://github.com/tensorflow/models.git

cd ..

docker run -it -p 8888:8888  -v /$(pwd):/test_files    gcr.io/tensorflow/tensorflow:latest-devel

root@...:~# cd /test_files/test_classify/models/tutorials/image/imagenet


#to download Inception Module and run internal test image

root@...:/test_files/test_classify/models/tutorials/image/imagenet#python classify_image.py

#this shows

Successfully downloaded inception-2015-12-05.tgz 88931400 bytes.

giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca (score = 0.89107)

indri, indris, Indri indri, Indri brevicaudatus (score = 0.00779)

lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens (score = 0.00296)

custard apple (score = 0.00147)

earthstar (score = 0.00117)


#to classify an image

root@....:/test_files/test_classify/models/tutorials/image/imagenet#

python classify_image.py --image_file /test_files/resources/classificationTestImage/bear.jpeg

#this shows

brown bear, bruin, Ursus arctos (score = 0.92993)

bottlecap (score = 0.00248)

American black bear, black bear, Ursus americanus, Euarctos americanus (score = 0.00157)

giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca (score = 0.00104)

sloth bear, Melursus ursinus, Ursus ursinus (score = 0.00103)

*********************************************************************
# Retrain Inception's Final Layer for New Categories

#Go to follwoing folder

cd chapter_09_folder_path

#and run docker

docker run -it -p 8888:8888  -v /$(pwd):/test_files    gcr.io/tensorflow/tensorflow:latest-devel

root@...:~# cd /tensorflow

root@...:/tensorflow# python tensorflow/examples/image_retraining/retrain.py \
--bottleneck_dir=/test_files/test_retrain/bottlenecks \
--how_many_training_steps 500 \
--model_dir=/test_files/test_retrain/inception \
--output_graph=/test_files/test_retrain/updated_retrained_graph.pb \
--output_labels=/test_files/test_retrain/updated_retrained_labels.txt \
--image_dir /test_files/resources/skin/c80

#shows

INFO:tensorflow:2017-09-07 15:08:12.871746: Step 499: Train accuracy = 100.0%

INFO:tensorflow:2017-09-07 15:08:12.871894: Step 499: Cross entropy = 0.051683

INFO:tensorflow:2017-09-07 15:08:12.924579: Step 499: Validation accuracy = 100.0% (N=100)

INFO:tensorflow:Final test accuracy = 100.0% (N=2)

root@...:/tensorflow# cd /test_files/test_retrain/

root@..:/test_files/test_retrain#  python detect_skin.py  /test_files/resources/skin/test/bengin_1.jpg

#shows

benign (score 0.96667)

malignant (score 0.03333)




#Delete these folder when you are done

root@...: rm -r  inception/  bottlenecks/ updated_retrained_*





