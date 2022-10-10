
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from tqdm.notebook import tqdm

# 損失関数値計算用
def eval_loss(loader, device, net, criterion):
  
    # DataLoaderから最初の1セットを取得する
    for images, labels in loader:
        break

    # デバイスの割り当て
    inputs = images.to(device)
    labels = labels.to(device)

    # 予測値の計算
    outputs = net(inputs)

    #  損失値の計算
    loss = criterion(outputs, labels)

    return loss

def new_fit(img ,device ,net ,outputs ,outputs4):
  imgs = []
  imgs.append(img)
  imgs = imgs.to(device)
  outputs2 = net(imgs)
  outputs3=outputs[i]+outputs2
  #1
  outputs4.append((outputs3[0]+outputs[1])/2)
  #2
  outputs4.append((outputs3[2]+outputs[3]+outputs[4]+outputs[5])/4)
  #3
  outputs4.append((outputs3[6]+outputs[7])/2)
  #4
  outputs4.append((outputs3[8]+outputs[9])/2)
  #5
  outputs4.append((outputs3[10]+outputs[11]+outputs[12]+outputs[13]+outputs[14])/5)
  #6
  outputs4.append((outputs3[15]+outputs[16])/2)
  #7
  outputs4.append((outputs3[17]+outputs[18]+outputs[19]+outputs[20])/4)
  #8
  outputs4.append((outputs3[21]+outputs[22])/2)
  #9
  outputs4.append((outputs3[23]+outputs[24])/2)
  #10
  outputs4.append((outputs3[25]+outputs[26])/2)
  #11
  outputs4.append((outputs3[27]+outputs[28]+outputs[29]+outputs[30]+outputs[31])/5)
  #12
  outputs4.append((outputs3[32]+outputs[33])/2)
  #13
  outputs4.append((outputs3[34]+outputs[35])/2)
  #14
  outputs4.append((outputs3[36]+outputs[37])/2)
  #15
  outputs4.append((outputs3[38]+outputs[39])/2)
  #16
  outputs4.append((outputs3[40]+outputs[41]+outputs[42]+outputs[43])/4)
  #17
  outputs4.append((outputs3[44]+outputs[45])/2)
  #18
  outputs4.append((outputs3[46]+outputs[47])/2)
  #19
  outputs4.append((outputs3[48]+outputs[49]+outputs[50]+outputs[51])/4)
  #20
  outputs4.append((outputs3[52]+outputs[53])/2)
  #21
  outputs4.append((outputs3[54]+outputs[55])/2)
  #22
  outputs4.append((outputs3[56]+outputs[57]+outputs[58]+outputs[59]+outputs[60]+outputs[61])/6)
  #23
  outputs4.append((outputs3[62]+outputs[63])/2)
  #24
  outputs4.append((outputs3[64]+outputs[65]+outputs[66]+outputs[67])/4)
  #25
  outputs4.append((outputs3[68]+outputs[69])/2)
  #26
  outputs4.append((outputs3[70]+outputs[71]+outputs[73]+outputs[74])/4)
  #27
  outputs4.append((outputs3[75]+outputs[76])/2)
  
  
  
  return outputs4

# 学習用関数
def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history ,test_transform):

    base_epochs = len(history)
  
    for epoch in range(base_epochs, num_epochs+base_epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        #訓練フェーズ
        net.train()
        count = 0

        for inputs, labels in tqdm(train_loader):
            count += len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # 予測計算
            outputs = net(inputs)

            # 損失計算
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            # 勾配計算
            loss.backward()

            # パラメータ修正
            optimizer.step()

            # 予測値算出
            predicted = torch.max(outputs, 1)[1]
            
            
            
           # predicted_2=torch.topk(outputs, 5, 1)
            

            # 正解件数算出
            train_acc += (predicted == labels).sum().item()

            # 損失と精度の計算
            avg_train_loss = train_loss / count
            avg_train_acc = train_acc / count

        #予測フェーズ
        net.eval()
        count = 0

        for inputs, labels in test_loader:
            count += len(labels)

            inputs = inputs.to(device)
            labels = labels.to(device)

            # 予測計算
            outputs = net(inputs)

            # 損失計算
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            #予測値算出
            predicted = torch.max(outputs, 1)[1]
            
            predicted_1=torch.topk(outputs,5,1)[0]
            #predicted_2=torch.topk(outputs, 5, 1)[1]
            
#original
            for i in range(len(labels)):
              list=[]
              rabels2=[]
              #1
              if 1<=rabels[i]<=2:
                rabels2.append(1)
                for k in range(1,3):
                  if k!=rabels[i]:
                    list.append(k)
                img = train_transform(Image.open('kisoko/test/010{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                  
              elif 3<=rabels[i]<=6:
                rabels2.append(2)
                for k in range(3,7):
                  if k!=rabels[i]:
                  list.append(7-k)
                img = train_transform(Image.open('kisoko/test/020{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                
              elif 7<=rabels[i]<=8:
                rabels2.append(3)
                for k in range(7,9):
                  if k!=rabels[i]:
                    list.append(9-k)
                img = train_transform(Image.open('kisoko/test/030{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                  
              elif 9<=rabels[i]<=10:
                rabels2.append(4)
                for k in range(9,11):
                  if k!=rabels[i]:
                    list.append(11-k)
                img = train_transform(Image.open('kisoko/test/040{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                  
              elif 11<=rabels[i]<=15:
                rabels2.append(5)
                for k in range(11,16):
                  if k!=rabels[i]:
                    list.append(16-k)
                img = train_transform(Image.open('kisoko/test/050{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                
              elif 16<=rabels[i]<=17:
                rabels2.append(6)
                for k in range(16,18):
                  if k!=rabels[i]:
                    list.append(18-k)
                img = train_transform(Image.open('kisoko/test/060{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                
                
              elif 18<=rabels[i]<=21:
                rabels2.append(7)
                for k in range(18,22):
                  if k!=rabels[i]:
                    list.append(22-k)
                img = train_transform(Image.open('kisoko/test/070{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                  
              elif 22<=rabels[i]<=23:
                rabels2.append(8)
                for k in range(22,24):
                  if k!=rabels[i]:
                    list.append(24-k)
                img = train_transform(Image.open('kisoko/test/080{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                
              elif 24<=rabels[i]<=25:
                rabels2.append(9)
                for k in range(24,26):
                  if k!=rabels[i]:
                    list.append(26-k)
                img = train_transform(Image.open('kisoko/test/090{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                  
              elif 26<=rabels[i]<=27:
                rabels2.append(10)
                for k in range(26,28):
                  if k!=rabels[i]:
                    list.append(28-k)
                img = train_transform(Image.open('kisoko/test/100{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                  
              elif 28<=rabels[i]<=32:
                rabels2.append(11)
                for k in range(28,33):
                  if k!=rabels[i]:
                    list.append(33-k)
                img = train_transform(Image.open('kisoko/test/110{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                
              elif 33<=rabels[i]<=34:
                rabels2.append(12)
                for k in range(33,35):
                  if k!=rabels[i]:
                    list.append(35-k)
                img = train_transform(Image.open('kisoko/test/120{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                    
                    
              elif 35<=rabels[i]<=36:
                rabels2.append(13)
                for k in range(35,37):
                  if k!=rabels[i]:
                    list.append(37-k)
                img = train_transform(Image.open('kisoko/test/130{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                  
              elif 37<=rabels[i]<=38:
                rabels2.append(14)
                for k in range(37,39):
                  if k!=rabels[i]:
                    list.append(39-k)
                img = train_transform(Image.open('kisoko/test/140{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                
              elif 39<=rabels[i]<=40:
                rabels2.append(15)
                for k in range(39,41):
                  if k!=rabels[i]:
                    list.append(41-k)
                img = train_transform(Image.open('kisoko/test/150{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                  
              elif 41<=rabels[i]<=44:
                rabels2.append(16)
                for k in range(41,45):
                  if k!=rabels[i]:
                    list.append(45-k)
                img = train_transform(Image.open('kisoko/test/160{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                  
              elif 45<=rabels[i]<=46:
                rabels2.append(17)
                for k in range(45,47):
                  if k!=rabels[i]:
                    list.append(47-k)
                img = train_transform(Image.open('kisoko/test/170{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                
              elif 47<=rabels[i]<=48:
                rabels2.append(18)
                for k in range(47,49):
                  if k!=rabels[i]:
                    list.append(49-k)
                img = train_transform(Image.open('kisoko/test/180{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                
                
              elif 49<=rabels[i]<=52:
                rabels2.append(19)
                for k in range(49,53):
                  if k!=rabels[i]:
                    list.append(53-k)
                img = train_transform(Image.open('kisoko/test/190{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                  
              elif 53<=rabels[i]<=54:
                rabels2.append(20)
                for k in range(53,55):
                  if k!=rabels[i]:
                    list.append(55-k)
                img = train_transform(Image.open('kisoko/test/200{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                
              elif 55<=rabels[i]<=56:
                rabels2.append(21)
                for k in range(55,57):
                  if k!=rabels[i]:
                    list.append(57-k)
                img = train_transform(Image.open('kisoko/test/210{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                  
              elif 57<=rabels[i]<=62:
                rabels2.append(22)
                for k in range(57,63):
                  if k!=rabels[i]:
                    list.append(63-k)
                img = train_transform(Image.open('kisoko/test/220{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                  
              elif 63<=rabels[i]<=64:
                rabels2.append(23)
                for k in range(63,65):
                  if k!=rabels[i]:
                    list.append(65-k)
                img = train_transform(Image.open('kisoko/test/230{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                
              elif 65<=rabels[i]<=68:
                rabels2.append(24)
                for k in range(65,69):
                  if k!=rabels[i]:
                    list.append(69-k)
                img = train_transform(Image.open('kisoko/test/240{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                 
              elif 69<=rabels[i]<=70:
                rabels2.append(25)
                for k in range(69,71):
                  if k!=rabels[i]:
                    list.append(71-k)
                img = train_transform(Image.open('kisoko/test/250{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                 
              elif 71<=rabels[i]<=74:
                rabels2.append(26)
                for k in range(71,75):
                  if k!=rabels[i]:
                    list.append(75-k)
                img = train_transform(Image.open('kisoko/test/260{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                      
              elif 75<=rabels[i]<=76:
                rabels2.append(27)
                for k in range(75,77):
                  if k!=rabels[i]:
                    list.append(77-k)
                img = train_transform(Image.open('kisoko/test/270{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
              
              outputs4=[]
              imgs = img.to(device)
              outputs2 = net(imgs)
              outputs3=outputs[i]+outputs2
              #1
              outputs4.append((outputs3[0]+outputs3[1])/2)
  #2
              outputs4.append((outputs3[2]+outputs3[3]+outputs3[4]+outputs3[5])/4)
  #3
              outputs4.append((outputs3[6]+outputs3[7])/2)
  #4
              outputs4.append((outputs3[8]+outputs3[9])/2)
  #5
              outputs4.append((outputs3[10]+outputs3[11]+outputs3[12]+outputs3[13]+outputs3[14])/5)
  #6
              outputs4.append((outputs3[15]+outputs3[16])/2)
  #7
              outputs4.append((outputs3[17]+outputs3[18]+outputs3[19]+outputs3[20])/4)
      #8
              outputs4.append((outputs3[21]+outputs3[22])/2)
  #9
              outputs4.append((outputs3[23]+outputs3[24])/2)
  #10
              outputs4.append((outputs3[25]+outputs3[26])/2)
  #11
              outputs4.append((outputs3[27]+outputs3[28]+outputs3[29]+outputs3[30]+outputs3[31])/5)
  #12
              outputs4.append((outputs3[32]+outputs3[33])/2)
  #13
              outputs4.append((outputs3[34]+outputs3[35])/2)
  #14
              outputs4.append((outputs3[36]+outputs3[37])/2)
  #15
              outputs4.append((outputs3[38]+outputs3[39])/2)
  #16
              outputs4.append((outputs3[40]+outputs3[41]+outputs3[42]+outputs3[43])/4)
  #17
              outputs4.append((outputs3[44]+outputs3[45])/2)
  #18
              outputs4.append((outputs3[46]+outputs3[47])/2)
  #19
              outputs4.append((outputs3[48]+outputs3[49]+outputs3[50]+outputs3[51])/4)
  #20
              outputs4.append((outputs3[52]+outputs3[53])/2)
  #21
              outputs4.append((outputs3[54]+outputs3[55])/2)
  #22
              outputs4.append((outputs3[56]+outputs3[57]+outputs3[58]+outputs3[59]+outputs3[60]+outputs3[61])/6)
  #23
              outputs4.append((outputs3[62]+outputs3[63])/2)
  #24
              outputs4.append((outputs3[64]+outputs3[65]+outputs3[66]+outputs3[67])/4)
  #25
              outputs4.append((outputs3[68]+outputs3[69])/2)
      #26
              outputs4.append((outputs3[70]+outputs3[71]+outputs3[73]+outputs3[74])/4)
  #27
              outputs4.append((outputs3[75]+outputs3[76])/2)
              
              predicted2 = torch.max(outputs4, 1)[1]
                  
                  
                  
                  
                  
                  
                  
            
#original
            

            #正解件数算出
            #val_acc += (predicted2 == labels2).sum().item()

            # 損失と精度の計算
            avg_val_loss = val_loss / count
            avg_val_acc = val_acc / count
    
        print (f'Epoch [{(epoch+1)}/{num_epochs+base_epochs}], loss: {avg_train_loss:.5f} acc: {avg_train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}')
        item = np.array([epoch+1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])
        history = np.vstack((history, item))
    return history



# 学習ログ解析
def evaluate_history(history):
  #損失と精度の確認
  print(f'初期状態: 損失: {history[0,3]:.5f} 精度: {history[0,4]:.5f}') 
  print(f'最終状態: 損失: {history[-1,3]:.5f} 精度: {history[-1,4]:.5f}' )

  num_epochs = len(history)
  if num_epochs < 10:
    unit = 1
  else:
    unit = num_epochs / 10

  # 学習曲線の表示 (損失)
  plt.figure(figsize=(9,8))
  plt.plot(history[:,0], history[:,1], 'b', label='訓練')
  plt.plot(history[:,0], history[:,3], 'k', label='検証')
  plt.xticks(np.arange(0,num_epochs+1, unit))
  plt.xlabel('繰り返し回数')
  plt.ylabel('損失')
  plt.title('学習曲線(損失)')
  plt.legend()
  plt.show()

  # 学習曲線の表示 (精度)
  plt.figure(figsize=(9,8))
  plt.plot(history[:,0], history[:,2], 'b', label='訓練')
  plt.plot(history[:,0], history[:,4], 'k', label='検証')
  plt.xticks(np.arange(0,num_epochs+1,unit))
  plt.xlabel('繰り返し回数')
  plt.ylabel('精度')
  plt.title('学習曲線(精度)')
  plt.legend()
  plt.show()


# イメージとラベル表示
def show_images_labels(loader, classes, net, device):

    # DataLoaderから最初の1セットを取得する
    for images, labels in loader:
        break
    # 表示数は50個とバッチサイズのうち小さい方
    n_size = min(len(images), 50)

    if net is not None:
      # デバイスの割り当て
      inputs = images.to(device)
      labels = labels.to(device)

      # 予測計算
      outputs = net(inputs)
      
      predicted = torch.max(outputs,1)[1]
      #images = images.to('cpu')

    # 最初のn_size個の表示
    plt.figure(figsize=(20, 15))
    for i in range(n_size):
        ax = plt.subplot(5, 10, i + 1)
        label_name = classes[labels[i]]
        # netがNoneでない場合は、予測結果もタイトルに表示する
        if net is not None:
          predicted_name = classes[predicted[i]]
          # 正解かどうかで色分けをする
          if label_name == predicted_name:
            c = 'k'
          else:
            c = 'b'
          ax.set_title(label_name + ':' + predicted_name, c=c, fontsize=20)
        # netがNoneの場合は、正解ラベルのみ表示
        else:
          ax.set_title(label_name, fontsize=20)
        # TensorをNumPyに変換
        image_np = images[i].numpy().copy()
        # 軸の順番変更 (channel, row, column) -> (row, column, channel)
        img = np.transpose(image_np, (1, 2, 0))
        # 値の範囲を[-1, 1] -> [0, 1]に戻す
        img = (img + 1)/2
        # 結果表示
        plt.imshow(img)
        ax.set_axis_off()
    plt.show()
    
