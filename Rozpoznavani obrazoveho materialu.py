import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
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

        self.who += self.lr * np.dot((skryta_chyba * vystupy_sv * (1 - vystupy_sv)), np.transpose(vstupy))

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

vstupni_neurony = 3
skryte_neurony = 3
vystupni_neurony = 3

# learning rate

ucici_koeficient = 0.3

# sestaveni neuronove site

n=neuralNetwork(vstupni_neurony,skryte_neurony,vystupni_neurony,ucici_koeficient)

# dotaz na neuronovou síť

ns=neuralNetwork(vstupni_neurony,skryte_neurony,vystupni_neurony,ucici_koeficient)

print(ns.dotaz([0,0,0]))
