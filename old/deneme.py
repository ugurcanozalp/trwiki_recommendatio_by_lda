import loaders

# Read Config YAML file.
cfg  = loaders.configloader()
tnlpt, zmbrk = loaders.nlplibloader()

txt = "Cengiz Han, 13. Yüzyılın başında Orta Asya'daki tüm göçebe bozkır kavimlerini birleştirerek bir ulus haline getirdi ve o ulusu Moğol siyasi kimliği çatısı altında topladı. Dünya tarihinin en büyük askeri liderlerinden biri olarak kabul edilen Cengiz Han, hükümdarlığı döneminde 1206-1227 arasında Kuzey Çin'deki Batı Xia ve Jin Hanedanı, Türkistan'daki Kara Hıtay, Maveraünnehir, Harezm, Horasan ve İran'daki Harzemşahlar ile Kafkasya'da Gürcüler, Deşt-i Kıpçak'taki Rus Knezlikleri ve Kıpçaklar ile İdil Bulgarları üzerine gerçekleştirilen seferler sonucunda Pasifik Okyanusu'ndan Hazar Denizi’ne ve Karadeniz'in kuzeyine kadar uzanan bir imparatorluk kurdu."
tnlpt.normalizeText(txt)
zmbrk.preprocesser(txt)
zmbrk.normalizeMultipleSentences(txt)