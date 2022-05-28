# CPS-3320-Final-Project W02 Group1
 Final NLP Project Sentimental Analysis
 In this project it totally has two approaches one is using Logistic Regression(by Bilu yao 1098213), this approach is initially used for generating the dict for the possiviteness of the words, but the accuarcy is not that accuracy in this approach, so that we decide to implements the next method with the help of neural network deep leaning(Chen zhiyang 1098489) with the help of Tensoflow.
# You Can find the data in the Data folder, totally two types sqlite and csv

# Here is the guide to run the python-sentilmental-analysis-deep-learning.py file

* Donwload Tensorflow via Command lie use pip install tensorflow
* use python python-sentilmental-analysis-deep-learning.py to run the code in cmd

>Totally 2 parameter will be desired inputted from the user: epoch and batch
>try-except will catch any input that is not integer
>then you can get the answer


* It may pop up something like:
> 2022-05-28 13:38:08.218751: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-05-28 13:38:08.219049: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
* which means you have download a GPU version of tensorflow it will be fine ignore it

* It may also pop up something like:
>C:\Users\27671\PycharmProjects\pythonProject5\3220Project.py:26: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  data['sentiments'] = data.Score.apply(lambda x: 0 if x in [1, 2] else 1)
C:\Users\27671\PycharmProjects\pythonProject5\3220Project.py:28: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

* which means that you have to use a lower version of python then the warnning will disappear. 
# You Can also use the jupter-deep-learning file if you directly put it directly in the jupyter.
This jupyter version is just a sample version that for testing the epoches and batcches, you can see the result after putting it in jupyter notebook. After that you can have a full view of the bloking of the code.

# Here is the guide to run the jupyter-logistic-regression file:
* Donwload sklearn extension in python by pip install sklearn
* Download nltk extension in python by pip install nltk
put the file directly in jupyter then it works. 



