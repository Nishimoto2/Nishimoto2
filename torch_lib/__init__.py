
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
from PIL import Image
import random

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

# 学習用関数
def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history ,train_transform):

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
            
            
            
           #predicted_2=torch.topk(outputs, 5, 1)
            

            # 正解件数算出
            train_acc += (predicted == labels).sum().item()

            # 損失と精度の計算
            avg_train_loss = train_loss / count
            avg_train_acc = train_acc / count

        #予測フェーズ
        net.eval()
        count = 0
        m = nn.Softmax(dim=1)
        

        for inputs, labels in test_loader:
            count += len(labels)

            inputs = inputs.to(device)
            labels = labels.to(device)

            # 予測計算
            outputs = m(net(inputs))

            # 損失計算
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            #予測値算出
            predicted = torch.max(outputs, 1)[1]
            
            #predicted_1=torch.topk(outputs,5,1)[0]
            #predicted_2=torch.topk(outputs, 5, 1)[1]
            labels2 = []
            outputs5 = []
            #print(labels)

            for i in range(len(labels)):
              list=[]
              if 0<=labels[i]<=1:
                labels2.append(0)
                for k in range(0,2):
                  if k!=labels[i]:
                    list.append(k+1)
                img = train_transform(Image.open('kisoko/test/010{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                  
              elif 2<=labels[i]<=5:
                labels2.append(1)
                for k in range(2,6):
                  if k!=labels[i]:
                    list.append(k-1)
                img = train_transform(Image.open('kisoko/test/020{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                
              elif 6<=labels[i]<=7:
                labels2.append(2)
                for k in range(6,8):
                  if k!=labels[i]:
                    list.append(k-5)
                img = train_transform(Image.open('kisoko/test/030{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                  
              elif 8<=labels[i]<=9:
                labels2.append(3)
                for k in range(8,10):
                  if k!=labels[i]:
                    list.append(k-7)
                img = train_transform(Image.open('kisoko/test/040{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                  
              elif 10<=labels[i]<=14:
                labels2.append(4)
                for k in range(10,15):
                  if k!=labels[i]:
                    list.append(k-9)
                img = train_transform(Image.open('kisoko/test/050{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))
                
              elif 15<=labels[i]<=16:
                labels2.append(5)
                for k in range(15,17):
                  if k!=labels[i]:
                    list.append(k-14)
                img = train_transform(Image.open('kisoko/test/060{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                
                
              elif 17<=labels[i]<=20:
                labels2.append(6)
                for k in range(17,21):
                  if k!=labels[i]:
                    list.append(k-16)
                img = train_transform(Image.open('kisoko/test/070{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))
                  
              elif 21<=labels[i]<=22:
                labels2.append(7)
                for k in range(21,23):
                  if k!=labels[i]:
                    list.append(k-20)
                img = train_transform(Image.open('kisoko/test/080{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))
   

                
              elif 23<=labels[i]<=24:
                labels2.append(8)
                for k in range(23,25):
                  if k!=labels[i]:
                    list.append(k-22)
                img = train_transform(Image.open('kisoko/test/090{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                  
              elif 25<=labels[i]<=26:
                labels2.append(9)
                for k in range(25,27):
                  if k!=labels[i]:
                    list.append(k-24)
                img = train_transform(Image.open('kisoko/test/100{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                  
              elif 27<=labels[i]<=31:
                labels2.append(10)
                for k in range(27,32):
                  if k!=labels[i]:
                    list.append(k-26)
                img = train_transform(Image.open('kisoko/test/110{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                
              elif 32<=labels[i]<=33:
                labels2.append(11)
                for k in range(32,34):
                  if k!=labels[i]:
                    list.append(k-31)
                img = train_transform(Image.open('kisoko/test/120{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                    
                    
              elif 34<=labels[i]<=35:
                labels2.append(12)
                for k in range(34,36):
                  if k!=labels[i]:
                    list.append(k-33)
                img = train_transform(Image.open('kisoko/test/130{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                  
              elif 36<=labels[i]<=37:
                labels2.append(13)
                for k in range(36,38):
                  if k!=labels[i]:
                    list.append(k-35)
                img = train_transform(Image.open('kisoko/test/140{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                
              elif 38<=labels[i]<=39:
                labels2.append(14)
                for k in range(38,40):
                  if k!=labels[i]:
                    list.append(k-37)
                img = train_transform(Image.open('kisoko/test/150{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                  
              elif 40<=labels[i]<=43:
                labels2.append(15)
                for k in range(40,44):
                  if k!=labels[i]:
                    list.append(k-39)
                img = train_transform(Image.open('kisoko/test/160{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                  
              elif 44<=labels[i]<=45:
                labels2.append(16)
                for k in range(44,46):
                  if k!=labels[i]:
                    list.append(k-43)
                img = train_transform(Image.open('kisoko/test/170{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                
              elif 46<=labels[i]<=47:
                labels2.append(17)
                for k in range(46,48):
                  if k!=labels[i]:
                    list.append(k-45)
                img = train_transform(Image.open('kisoko/test/180{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                
                
              elif 48<=labels[i]<=51:
                labels2.append(18)
                for k in range(48,52):
                  if k!=labels[i]:
                    list.append(k-47)
                img = train_transform(Image.open('kisoko/test/190{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                  
              elif 52<=labels[i]<=53:
                labels2.append(19)
                for k in range(52,54):
                  if k!=labels[i]:
                    list.append(k-51)
                img = train_transform(Image.open('kisoko/test/200{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                
              elif 54<=labels[i]<=55:
                labels2.append(20)
                for k in range(54,56):
                  if k!=labels[i]:
                    list.append(k-53)
                img = train_transform(Image.open('kisoko/test/210{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                  
              elif 56<=labels[i]<=61:
                labels2.append(21)
                for k in range(56,62):
                  if k!=labels[i]:
                    list.append(k-55)
                img = train_transform(Image.open('kisoko/test/220{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                  
              elif 62<=labels[i]<=63:
                labels2.append(22)
                for k in range(62,64):
                  if k!=labels[i]:
                    list.append(k-61)
                img = train_transform(Image.open('kisoko/test/230{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                
              elif 64<=labels[i]<=67:
                labels2.append(23)
                for k in range(64,68):
                  if k!=labels[i]:
                    list.append(k-63)
                img = train_transform(Image.open('kisoko/test/240{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                 
              elif 68<=labels[i]<=69:
                labels2.append(24)
                for k in range(68,70):
                  if k!=labels[i]:
                    list.append(k-67)
                img = train_transform(Image.open('kisoko/test/250{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                 
              elif 70<=labels[i]<=73:
                labels2.append(25)
                for k in range(70,74):
                  if k!=labels[i]:
                    list.append(k-69)
                img = train_transform(Image.open('kisoko/test/260{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))

                      
              elif 74<=labels[i]<=75:
                labels2.append(26)
                for k in range(74,76):
                  if k!=labels[i]:
                    list.append(k-73)
                img = train_transform(Image.open('kisoko/test/270{}/IMG_{} 小.jpeg'.format(random.choice(list),random.randrange(1,4))))
                img_batch = img[None]
                img_batch = img_batch.to(device)
                outputs2 = m(net(img_batch))
              
              #print(outputs2)
              #print('///')
              #print(outputs[i,:])
              outputs3 = outputs2[0,:]
              #print(outputs3)
              outputs4 = outputs[i,:] + outputs3
              #print(outputs4)
              outputs4 = outputs4.to('cpu').detach().numpy()
              #print(outputs4)
              #print(labels2)
              
              
              
              outputs5.append((outputs4[0]+outputs4[1])/2)
              outputs5.append((outputs4[2]+outputs4[3]+outputs4[4]+outputs4[5])/4)
              outputs5.append((outputs4[6]+outputs4[7])/2)
              outputs5.append((outputs4[8]+outputs4[9])/2)
              outputs5.append((outputs4[10]+outputs4[11]+outputs4[12]+outputs4[13]+outputs4[14])/5)
              outputs5.append((outputs4[15]+outputs4[16])/2)
              outputs5.append((outputs4[17]+outputs4[18]+outputs4[19]+outputs4[20])/4)
              outputs5.append((outputs4[21]+outputs4[22])/2)
              outputs5.append((outputs4[23]+outputs4[24])/2)
              outputs5.append((outputs4[25]+outputs4[26])/2)
              outputs5.append((outputs4[27]+outputs4[28]+outputs4[29]+outputs4[30]+outputs4[31])/5)
              outputs5.append((outputs4[32]+outputs4[33])/2)
              outputs5.append((outputs4[34]+outputs4[35])/2)
              outputs5.append((outputs4[36]+outputs4[37])/2)
              outputs5.append((outputs4[38]+outputs4[39])/2)
              outputs5.append((outputs4[40]+outputs4[41]+outputs4[42]+outputs4[43])/4)
              outputs5.append((outputs4[44]+outputs4[45])/2)
              outputs5.append((outputs4[46]+outputs4[47])/2)
              outputs5.append((outputs4[48]+outputs4[49]+outputs4[50]+outputs4[51])/4)
              outputs5.append((outputs4[52]+outputs4[53])/2)
              outputs5.append((outputs4[54]+outputs4[55])/2)
              outputs5.append((outputs4[56]+outputs4[57]+outputs4[58]+outputs4[59]+outputs4[60]+outputs4[61])/6)
              outputs5.append((outputs4[62]+outputs4[63])/2)  
              outputs5.append((outputs4[64]+outputs4[65]+outputs4[66]+outputs4[67])/4)
              outputs5.append((outputs4[68]+outputs4[69])/2)  
              outputs5.append((outputs4[70]+outputs4[71]+outputs4[72]+outputs4[73])/4)
              outputs5.append((outputs4[74]+outputs4[75])/2) 
              #print(outputs5)
              
              
                  
                  
                  
            
#original
            
            outputs5 = torch.tensor(outputs5).float()
            labels2 = torch.tensor(labels2).long()
            outputs5 = outputs5.view(2,-1)
            #print(outputs5)
            #loss = criterion(outputs5, labels2)
            #val_loss += loss.item()
            predicted2 = torch.max(outputs5, 1)[1]
            #print(predicted2)
            #print(labels2)
            #正解件数算出
            val_acc += (predicted2 == labels2).sum().item()

            # 損失と精度の計算
            #avg_val_loss = val_loss / count
            avg_val_acc = val_acc / count
    
        print (f'Epoch [{(epoch+1)}/{num_epochs+base_epochs}], loss: {avg_train_loss:.5f} acc: {avg_train_acc:.5f} val_acc: {avg_val_acc:.5f}')
        item = np.array([epoch+1, avg_train_loss, avg_train_acc, avg_val_acc])
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
    
