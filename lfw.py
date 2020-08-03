import os
from os import path
from verification import *
import time
import numpy as np
import matplotlib.pyplot as plt

model_path = path.join('model','facenet_keras.h5')
model = get_model(model_path)

os.chdir(os.path.join('dataset','lfw'))
issame_pair = []
diff_people = []

def evaluate(same, diff):
  TPR = []
  FPR = []
  TP = []
  TN = []
  acc = []
  for t in np.arange(0,1,0.001):                         # 'same' = true ; 'different' = false
    true_pos = np.less_equal(np.asarray(same),t).sum()   # #pairs classified 'same' in same list
    true_neg = np.greater(np.asarray(diff),t).sum()      # #pairs classified 'different in diff list
    false_pos = np.less_equal(np.asarray(diff),t).sum()  # #pairs classified 'same' in diff list
    false_neg = np.greater(np.asarray(same),t).sum()     # #pairs classified 'different' in same list
    true_positive_rate = true_pos / len(same)
    specificity = true_neg / len(diff)
    false_positive_rate = 1 - specificity
    TP.append(true_pos)
    TN.append(true_neg)
    TPR.append(true_positive_rate)
    FPR.append(false_positive_rate)
    acc.append((true_pos + true_neg)/(len(same) + len(diff)))

  plt.plot(np.arange(0,1,0.001),TP, label='TP')
  plt.plot(np.arange(0,1,0.001),TN, label='TN')
  plt.xlabel('Threshold')
  plt.legend()
  plt.show()


  plt.plot(FPR,TPR)
  plt.xlabel('FPR')
  plt.ylabel('TPR')
  plt.title('ROC Curve')
  plt.show()

  opt_threshold = np.argmax(np.asarray(TPR)-np.asarray(FPR))

  return {'tpr':TPR, 'fpr':FPR, 'optimal-threshold':opt_threshold, 'optimal-accuracy':acc[opt_threshold]}  

if __name__ == '__main__':
  with open("pairs.txt","r") as f:
    f.readline()
    for s in f.readlines():
      words = s.split()
      if len(words)==3 :    
        main_dir = words[0]
        img1_path = words[0]+("_000" if len(words[1])==1 else "_00" if len(words[1])==2 else "_0") + words[1]+".jpg"
        img2_path = words[0]+("_000" if len(words[2])==1 else "_00" if len(words[2])==2 else "_0") + words[2]+".jpg"
        img1_path = os.path.join(main_dir,img1_path)
        img2_path = os.path.join(main_dir,img2_path)
        issame_pair.append((img1_path,img2_path))    
      elif len(words)==4 :    
        main_dir1 = words[0]
        img1_path = words[0]+("_000" if len(words[1])==1 else "_00" if len(words[1])==2 else "_0") + words[1]+".jpg"
        main_dir2 = words[2]
        img2_path = words[2]+("_000" if len(words[3])==1 else "_00" if len(words[3])==2 else "_0") + words[3]+".jpg"
        img1_path = os.path.join(main_dir1,img1_path)
        img2_path = os.path.join(main_dir2,img2_path)
        diff_people.append((img1_path,img2_path))

  print("Length of issame_pair:", len(issame_pair))
  print("Length of diff_people:", len(diff_people))      

  same = []       
  diff = []
  times = []
  errors = []

  for pair in issame_pair:
      start_time = time.time()
      if path.exists(pair[0]) and path.exists(pair[1]):
          embeddings = model.predict(prepare(pair[0],pair[1]))
          same.append(cosine_dist(embeddings[0],embeddings[1]))
          times.append(time.time() - start_time)
      else :
          errors.append(pair)   

  for pair in diff_people:
      start_time = time.time()
      if path.exists(pair[0]) and path.exists(pair[1]):
          embeddings = model.predict(prepare(pair[0],pair[1]))
          diff.append(cosine_dist(embeddings[0],embeddings[1]))
          times.append(time.time() - start_time)
      else:
          errors.append(pair)

  print("Length of same", len(same))
  print("Length of diff", len(diff))

  result = evaluate(same,diff)

  print("Accuracy", result['optimal-accuracy'], "Time Required", np.mean(times),'+/-',np.std(times))
  if(len(errors)):
    print("Following pairs didn't exist")
    print(errors)



        



