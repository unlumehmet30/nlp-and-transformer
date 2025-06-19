# -*- coding: utf-8 -*-
"""
Created on Sat May 24 17:34:37 2025

@author: mhmtn
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#data created by chatgpt
text = [
    "Bugün hava gerçekten çok güzel.",
    "Sabah kahvemi içmeden kendime gelemiyorum.",
    "Dün gece çok geç yattım, uyanmak zor oldu.",
    "Akşam ne yemek yapsam bir türlü karar veremiyorum.",
    "Film izlemek için güzel bir gün.",
    "İşe gitmek bugün hiç içimden gelmiyor.",
    "Yeni bir kitaba başladım, çok sürükleyici.",
    "Arkadaşlarla buluşmak iyi geldi.",
    "Hafta sonunu dört gözle bekliyorum.",
    "Bugün dışarı çıkmak istemiyorum.",
    "Sınavlar yaklaşıyor, çalışmam gerek.",
    "Yorgunum ama mutlu hissediyorum.",
    "Markete gitmem lazım, evde hiçbir şey kalmadı.",
    "Telefonumun şarjı yine bitmek üzere.",
    "Kahvaltı yapmadan dışarı çıkmam.",
    "Yeni tarif denedim, sonuç harika oldu.",
    "Biraz müzik dinlemek iyi gelir şimdi.",
    "Yarın sabah erken kalkmam gerekiyor.",
    "Uykum var ama işlerim bitmedi.",
    "Hadi bir kahve molası verelim.",
    "Bugün işlerim çok yoğundu.",
    "Yeni bir şeyler öğrenmek her zaman heyecan verici.",
    "Dışarısı çok kalabalık, çıkasım yok.",
    "Yemek siparişi vermek daha kolay geliyor.",
    "Dün çok güzel bir rüya gördüm.",
    "Biraz temiz hava almak istiyorum.",
    "Kedim bütün gece uyutmadı.",
    "Bugün hiçbir şey yapmak istemiyorum.",
    "Yeni sezona başlayan diziyi izledin mi?",
    "Spora başlamak istiyorum ama bir türlü başlayamıyorum.",
    "Sürekli ertelemekten sıkıldım artık.",
    "Bazen her şeyi bırakıp gitmek istiyorum.",
    "Sessiz bir ortamda çalışmak daha kolay.",
    "Bugün yağmur yağsa ne güzel olurdu.",
    "Sabahları yürüyüş yapmak çok iyi geliyor.",
    "Telefonum sürekli çalıyor, dinlenemiyorum.",
    "Bir kahve içip rahatlamam lazım.",
    "Kütüphane sessizliği tam benlik.",
    "Dışarıda rüzgar var, montsuz çıkılmaz.",
    "Bugün pazartesi sendromu yaşadım.",
    "Yeni ayakkabılar ayağımı vurdu.",
    "Bir tatil planı yapmak istiyorum.",
    "Hava çok sıcak, evden çıkmak istemiyorum.",
    "Bugün ne giysem bir türlü karar veremedim.",
    "Arkadaşımın doğum günü yaklaşıyor.",
    "İnternette dolaşırken saat nasıl geçti anlamadım.",
    "Ev işleri hiç bitmiyor.",
    "Yarın ne yapsam acaba?",
    "Yeni tarif denedim ama hiç güzel olmadı.",
    "Sabah trafiği gerçekten çok can sıkıcı.",
    "Bugün biraz kitap okuyacağım.",
    "Uzun zamandır dışarı çıkmadım.",
    "Canım tatlı bir şeyler çekiyor.",
    "Film izlerken uyuyakalmışım.",
    "Bugün temizlik günü ilan ettim.",
    "Kargom hala gelmedi, çok sinirliyim.",
    "Bugün enerjim yerinde.",
    "Ruh halim biraz dalgalı bugün.",
    "Müzik açınca modum hemen yükseliyor.",
    "Yarın için plan yapmadım henüz.",
    "Bugün annemle uzun uzun konuştum.",
    "Yeni bir dil öğrenmek istiyorum.",
    "Gün çok çabuk geçti, farkında bile değilim.",
    "Sıcak bir çorba olsa şimdi çok iyi olurdu.",
    "Film izlerken patlamış mısır şart.",
    "Çok çalıştım, artık biraz dinlenmeliyim.",
    "Uykum bir türlü gelmiyor.",
    "Yarın hava nasıl olacak acaba?",
    "Sürekli telefonuma bakmaktan yoruldum.",
    "Bugün güzel bir gün geçirmek istiyorum.",
    "Biraz açık hava fena olmazdı.",
    "Yine geç kaldım, her zamanki gibi.",
    "Kahve dükkanında oturmak huzur veriyor.",
    "Yeni diziler keşfetmek keyifli oluyor.",
    "Hafta sonu için güzel planlarım var.",
    "Bugün dışarıda yürüyüş yapmak istiyorum.",
    "Biraz sosyal medyadan uzaklaşmak lazım.",
    "Kafam çok dolu, biraz düşünmem gerek.",
    "Rahat bir kıyafet giymek istiyorum.",
    "Arkadaşımın tavsiye ettiği filmi izledim.",
    "Bugün biraz tembellik yapmak istiyorum.",
    "Uzun süredir kitap okumuyorum.",
    "Bir tatlı molası verebiliriz.",
    "Kafamı dağıtmak için bir şeyler yapmalıyım.",
    "Bugün işlerim beklediğimden hızlı bitti.",
    "Sıcaktan bunaldım.",
    "Yemekten sonra bir yürüyüş iyi olur.",
    "Yeni bir hobi edinmek istiyorum.",
    "Bugün kendimi motive hissetmiyorum.",
    "Biraz sessizlik fena olmazdı.",
    "Küçük şeylerle mutlu olmayı seviyorum.",
    "Bazen hiçbir şey yapmak istemiyorum.",
    "Bu sabah erkenden uyandım.",
    "Bugün dolu dolu geçti.",
    "Arkadaşım beni kahveye davet etti.",
    "Kahve içmeden güne başlayamam.",
    "Biraz müzik dinlemek ruhuma iyi gelir.",
    "Yolda yürürken eski bir arkadaşımı gördüm.",
    "Bugün kendime vakit ayırmak istiyorum.",
    "Yeni bir şeyler denemek istiyorum.",
    "Kahvaltı yapmadan güne başlayamam.",
    "Dün gece çok güzel bir film izledim.",
    "Yorgunluktan gözlerim kapanıyor.",
    "Bugün biraz resim yapacağım.",
    "Akşam yürüyüşleri çok keyifli.",
    "Yeni hedefler belirlemem lazım.",
    "Sakin bir pazar günü geçiriyorum.",
    "Bugün yeni insanlarla tanıştım.",
    "Yolda kitap okuyan birini görünce mutlu oluyorum.",
    "Sıcak çay içmek bana huzur veriyor.",
    "Bugün biraz temizlik yapacağım.",
    "Kalabalıktan uzak bir yer arıyorum.",
    "Yolda yürürken müzik dinlemeyi seviyorum.",
    "Bugün kendime bir ödül verdim.",
    "Yeni bir diziye başladım.",
    "Bütün gün evde kaldım.",
    "Dışarıda hava çok güzel, biraz gezmek istiyorum.",
    "Kitap okumak bana iyi geliyor.",
    "Kahvemi aldım, şimdi işlerime dönebilirim.",
    "Biraz yürüyüş yapmak istiyorum.",
    "Dünden kalan işleri halletmem gerek.",
    "Bugün erken yatacağım.",
    "Telefonumla çok vakit geçiriyorum.",
    "Bugün güzel haberler aldım.",
    "Evde sessizlik hakim.",
    "Kendime küçük bir tatil planlıyorum.",
    "Yeni şeyler öğrenmek beni mutlu ediyor.",
    "Sade kahve içmeyi seviyorum.",
    "Sabah yürüyüşü bana enerji veriyor.",
    "Bugün hiç kimseyle konuşmadım.",
    "Sürekli aynı şeyleri yapmak sıkıcı oluyor.",
    "Biraz dinlenmek bana iyi gelecek.",
    "Hafif bir yemek yesem yeter.",
    "Bugün moralim yerinde.",
    "Biraz dışarı çıkıp hava almak lazım.",
    "Küçük şeylerden mutlu olmayı öğrendim.",
    "Bir fincan kahve eşliğinde kitap okuyorum.",
    "Bugün sakin bir gün geçirdim.",
    "Müzik ruhun gıdası gerçekten.",
    "Yarın sabah kahvaltıya dışarı çıkalım mı?",
    "Kendime biraz zaman ayırmam gerekiyordu.",
    "Bugün pencereden dışarıyı izlemekle yetindim.",
    "Akşam için güzel bir planım var.",
    "Sadece sessizlik istiyorum.",
    "Yine geç uyandım.",
    "Bugün her şey yolunda gitti.",
    "Hafif bir yürüyüş iyi gelir.",
    "Kafam çok dolu, biraz temiz hava almalıyım.",
    "Bir film açıp battaniyeye sarılmak istiyorum.",
    "Bugün hiçbir şey yapmadan geçsin istiyorum.",
    "Hayat bazen gerçekten çok garip oluyor.",
    "Sıcacık bir kahveye kim hayır der ki?",
    "Biraz dışarı çıkıp dolaşmak istiyorum.",
    "Müzik açınca bütün dertlerim uçup gidiyor.",
    "Bugün tamamen kendime ait bir gün olsun istiyorum.",
    "Kafamı dağıtmak için biraz temizlik yapacağım.",
    "Güzel bir gün geçirmek istiyorum.",
    "Telefonumu sessize aldım, biraz huzur zamanı.",
    "Bugün sadece huzur istiyorum.",
    "Bir tatlı alıp keyif yapmak istiyorum.",
    "Yarın için heyecanlıyım.",
    "Yemek yapmak bazen gerçekten terapi gibi.",
    "Güzel bir kahveyle güne başlamak harika.",
    "Dışarısı çok sessiz, huzur verici.",
    "Bugün kendime iyi davranacağım.",
    "Dışarıda yağmur yağıyor, çok romantik.",
    "Güne güzel bir şarkıyla başladım.",
    "Biraz yalnız kalmak istiyorum.",
    "Kahvemi aldım, şimdi rahatladım.",
    "Bugün günlerden neydi unuttum.",
    "Yataktan çıkmak zor geliyor.",
    "Bir tatil hiç fena olmazdı.",
    "Kendimi dinlemeye ihtiyacım var.",
    "Bugün pozitif hissediyorum.",
    "Garson bizi uzun süre bekletti.",
    "Yemekler sıcacık geldi, çok memnun kaldık.",
    "Tatlı çok kötüydü, yiyemedim.",
    "Mekan çok şirin ve keyifliydi.",
    "Servis rezaletti, hiç ilgilenmediler.",
    "Garsonlar çok ilgiliydi, teşekkür ederiz.",
    "Salata bayat ve tatsızdı.",
    "Kahvaltı harikaydı, çeşit çok fazlaydı.",
    "Yemekleri hiç beğenmedik, çok tuzluydu.",
    "Porsiyonlar doyurucu ve güzeldi.",
    "Masalar kirliydi, hiç temizlenmemiş.",
    "Mekan oldukça ferah ve rahattı.",
    "Garson siparişi karıştırdı, yanlış yemek geldi.",
    "Hizmetten çok memnun kaldık.",
    "Et pişmemişti, çiğdi resmen.",
    "Müzik çok hoştu, ambiyans mükemmeldi.",
    "Tatlı bayattı ve tadı kötüydü.",
    "İkram edilen çay çok hoştu.",
    "Rezervasyonumuz olmasına rağmen yer yoktu.",
    "Sunum çok güzeldi, fotoğraf çektim.",
    "Lavabo çok pisti, rahatsız oldum.",
    "Yemekten sonra tatlı ikramı çok hoştu.",
    "Çalışanlar çok kaba davrandı.",
    "Tatlıyı çok beğendim, tarifini istedim.",
    "İçecekler gazsızdı, tadı kaçmıştı.",
    "Personel çok yardımcıydı, teşekkür ederiz.",
    "Menü çok sade ama yeterliydi.",
    "Havalandırma kötüydü, içerisi havasızdı.",
    "Mekan aşırı gürültülüydü.",
    "Servis hızlı ve düzenliydi.",
    "Sandalyeler konforluydu, rahat ettik.",
    "Hesap yanlış geldi, düzeltmek zorunda kaldık.",
    "Kebaplar mükemmeldi, tekrar geleceğim.",
    "Garsonlar ilgisizdi, defalarca seslendik.",
    "Tatlıların görüntüsü harikaydı.",
    "Çatal ve bıçaklar lekeliydi.",
    "Müzik sesi çok yüksekti.",
    "Yemekler özenle hazırlanmıştı.",
    "Garsonun tavsiyesi çok yerindeydi.",
    "İçecek çok ılıktı, buz istemek zorunda kaldık.",
    "Tatlıdan böcek çıktı, şok oldum.",
    "Çalışanlar çok nazik ve anlayışlıydı.",
    "Porsiyonlar küçük ama çok lezzetliydi.",
    "Tuvalet tertemizdi, çok şaşırdım.",
    "Garson bize hiç bakmadı.",
    "Sunum yaratıcı ve çok şıktı.",
    "Yemek beklediğimizden hızlı geldi.",
    "Yemekler soğuktu, mikrodalga mı kullanıldı acaba?",
    "Kahvaltı taze ve doyurucuydu.",
    "Tatlılar tatsızdı, hiç beğenmedik."
]

#preprocessing
tokenizer=Tokenizer()
tokenizer.fit_on_texts(text)
total_words=len(tokenizer.word_index)+1

input_sequences=[]
for text in text:
    token_list=tokenizer.texts_to_sequences([text])[0]
    for i in range(1,len(token_list)):
        n_gram_sequences=token_list[:i+1]
        input_sequences.append(n_gram_sequences)
        
max_seq=max(len(x) for x in input_sequences)

input_sequences=pad_sequences(input_sequences,maxlen=max_seq,padding="pre")

x,y=input_sequences[:,:-1],input_sequences[:,-1]
#model compile train evaulate
y=tf.keras.utils.to_categorical(y,num_classes=total_words)
model=Sequential()
model.add(Embedding(total_words,50,input_length=x.shape[1]))
model.add(LSTM(100,return_sequences=False))
model.add(Dense(total_words,activation="softmax"))
model.compile(optimizer="adam",loss="categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x,y,epochs=100,verbose=1)

def generate_text(seed_text,n_word): 
    for _ in range(n_word):
        token_list=tokenizer.texts_to_sequences([seed_text])[0]
        token_list=pad_sequences([token_list],maxlen=max_seq-1,padding="pre")
        predicted=model.predict(token_list,verbose=0)
        predicted_word_index=np.argmax(predicted,axis=-1)
        predicted_word=tokenizer.index_word[predicted_word_index[0]]
        seed_text=seed_text+" "+predicted_word
    return seed_text
seed_text="Bugün hava"
seed_text=generate_text(seed_text, 4)
print(f"text:{seed_text}")
































