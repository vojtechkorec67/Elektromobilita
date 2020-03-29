import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scipy.special

warnings.filterwarnings("ignore")


# Neuronová síť, definování třídy

class neuralNetwork :

    # inicializace neuronové sítě
    def __init__(self, vstupni_neurony, skryte_neurony, vystupni_neurony, ucici_koeficient):

        self.inodes = vstupni_neurony
        self.hnodes = skryte_neurony
        self.onodes = vystupni_neurony

        # váhy uvnitř matic mají tvar Wij, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc

        # jednoduche nastaveni vah, pomoci nahodneho nastaveni vah v rozmezi -0.5 a 0.5
        # puvodne byly nastaveny v rozmezí (-1,1), pomoci odecteni  "-0.5" dostane vahy v nahodnem rozdeleni v rozmezi (-0.5,0.5)


        # self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        # self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)

        # alternativni a sofistikovanejsi nastaveni vah

        self.wih = (np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        # learning rate
        self.lr= ucici_koeficient

        # aktivační funkce
        #self.aktivacni_funkce = lambda x: scipy.special.expit(x)
        self.aktivacni_funkce = lambda x: (np.exp(x)) / (np.exp(x) + 1)

        pass

    # trénování neuronové sítě
    def train(self,list_vstupu,cilove_hodnoty):

        # konvertování listu se vstupy do 2-rozměrné numpy matice. ".T" : transpozice 2-rozměrné matice
        vstupy=np.array(list_vstupu,ndmin=2).T
        cilove_hodnoty=np.array(cilove_hodnoty,ndmin=2).T

        # výpočet signalu vstupujícího do skryté vrstvy

        vstupy_sv = np.dot(self.wih, vstupy)

        # výpočet signálu vystupujícího ze skryté vrstvy

        vystupy_sv = self.aktivacni_funkce(vstupy_sv)

        # vypocet signalu vstupujiciho do vystupni vrstvy

        vstupy_vv = np.dot(self.who,vystupy_sv)

        # vypocet signalu vystupujiciho z vystupni vrstvy

        vystupy_vv = self.aktivacni_funkce(vstupy_vv)

        # chyba sítě na výstupu (cílové hodnoty - predikované hodnoty)

        vystupni_chyba = cilove_hodnoty - vystupy_vv

        # chyba skryté vrstvy

        skryta_chyba = np.dot(self.who.T, vystupni_chyba)

        # aktualizace vah jednotlivych synapsi mezi skrytou a vystupni vrstvou

        self.who += self.lr * np.dot((vystupni_chyba * vystupy_vv * (1.0 - vystupy_vv)),np.transpose(vystupy_sv))

        # aktualizace vah jednotlivych synapsi mezi vstupni a skrytou vrstvou

        self.wih += self.lr * np.dot((skryta_chyba * vystupy_sv * (1.0 - vystupy_sv)), np.transpose(vstupy))

        pass

    # dotazování neuronové sítě
    def dotaz(self,list_vstupu):

        # konvertování listu se vstupy do 2-rozměrné numpy matice. ".T" : transpozice 2-rozměrné matice
        vstupy = np.array(list_vstupu, ndmin=2).T

        #  výpočet signalu vstupujiciho do skryte vrstvy
        vstupy_sv = np.dot(self.wih, vstupy)

        # výpočet signalu vystupujiciho ze skryte vrstvy
        vystupy_sv = self.aktivacni_funkce(vstupy_sv)

        # vypocet signalu vstupujiciho do vystupni vrstvy
        vstupy_vv = np.dot(self.who, vystupy_sv)

        # vypocet signalu vystupujiciho z vystupni vrstvy
        vystupy_vv = self.aktivacni_funkce(vstupy_vv)

        return vystupy_vv

        pass

# pocet vstupu, skrytych neuronu, vystupnich neuronu

vstupni_neurony = 784
skryte_neurony = 100
vystupni_neurony = 10

# ucici koeficienty
ucici_koeficient = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# presnost predikci pro ruznych ucicich koeficientech
procentualni_presnost_predikci = []

# realizace neuronovych siti pro jednotlive ucici koeficienty
for i in ucici_koeficient:
  hodnoceni_site = []

# sestaveni neuronove site

  ns=neuralNetwork(vstupni_neurony,skryte_neurony,vystupni_neurony,i)

# nacteni trenovacich dat s obrazovym materialem
  trenovaci_data=open("C:\\Users\\korec\\PycharmProjects\\Elektromobilita\\mnist_train_full_size.csv","r")
  trenovaci_data_list = trenovaci_data.readlines()

# trenovani neuronove site
  for zaznam in trenovaci_data_list :
    # rozdeleni zaznamu pomoci ","
    hodnoty_obrazu = zaznam.split(",")

    # normalizování hodnot vstupních proměných do rozmezi (0.1 , 1.00)
    normalizovane_hodnoty_vstupu = (np.asfarray(hodnoty_obrazu[1:]) / 255.0 * 0.99) + 0.01

    # cílové matice hodnot pro trenovani neuronove site
    cilova_matice_hodnot_vystupu = np.zeros(vystupni_neurony) + 0.01

    # hodnoty[0] jsou klasifikace pro kazdy zaznam hodnot (pro kazdy pripad/pro kazdy radek)
    cilova_matice_hodnot_vystupu[int(hodnoty_obrazu[0])] = 0.99

    # trenovani neuronove site
    ns.train(normalizovane_hodnoty_vstupu,cilova_matice_hodnot_vystupu)

    pass

# nacteni testovacich dat
  testovaci_data = open("C:\\Users\\korec\\PycharmProjects\\Elektromobilita\\mnist_test_full_size.csv","r")
  testovaci_data_list = testovaci_data.readlines()

# hodnoceni site
  #hodnoceni_site = []

# testovani neuronove site
  for zaznam in testovaci_data_list :


  # rozdeleni zaznamu pomoci ","
   hodnoty_obrazu = zaznam.split(",")

  # correct answer
   spravna_odpoved = int(hodnoty_obrazu[0])
   #print(spravna_odpoved,"spravna odpoved")

  # normalizování hodnot vstupních proměných do rozmezi (0.1 , 1.00)
   vstupy = (np.asfarray(hodnoty_obrazu[1:]) / 255.0*0.99) + 0.01

  # dotazovani neuronove site
   vystupy_site = ns.dotaz(vstupy)

  # index nejvyssi hodnoty odpovida label
   label = np.argmax(vystupy_site)
   #print(label, "predikce NS")

  # pokud se "spravna odpoved" = "label" pak 1, jinak 0
   if (label == spravna_odpoved) :
     hodnoceni_site.append(1)
   else:
     hodnoceni_site.append(0)

  #presnost_predikci.append(hodnoceni_site)
  procentualni_presnost = (np.array(hodnoceni_site).sum() / len(hodnoceni_site))
  procentualni_presnost_predikci.append(procentualni_presnost)

pass

# graficke znazorneni presnosti predikci

sns.lineplot(x=ucici_koeficient,y=procentualni_presnost_predikci)
plt.title("Presnost site v zavislosti na ucicim koeficientu")
plt.xlabel("ucici koeficient")
plt.ylabel("presnost site")
plt.yticks(np.arange(0.7,0.99,0.01))
plt.savefig("C:\\Users\\korec\\PycharmProjects\\Elektromobilita\\presnost_v_zavislosti_na_ucicim_koeficientu.png")
plt.show()















