import pandas as pd
import matplotlib.pyplot as plt

def plot1():
    df = pd.read_csv('waz1_rozrz_stan.csv')
    plt.figure(figsize=(14, 8))
    plt.plot(df.iloc[1:, 1:])
    plt.title("Wąż numer 1, rozrzeszony stan")
    plt.xlabel('Numer gry')
    plt.ylabel('Wynik')
    #plt.show()
    plt.savefig("waz1_rozrz_stan")

    df = pd.read_csv('waz2_rozrz_stan.csv')
    plt.figure(figsize=(14, 8))
    plt.plot(df.iloc[1:, 1:])
    plt.title("Wąż numer 2, rozrzeszony stan")
    plt.xlabel('Numer gry')
    plt.ylabel('Wynik')
    #plt.show()
    plt.savefig("waz2_rozrz_stan")

    df = pd.read_csv('waz3_rozrz_stan.csv')
    plt.figure(figsize=(14, 8))
    plt.plot(df.iloc[1:, 1:])
    plt.title("Wąż numer 3, rozrzeszony stan")
    plt.xlabel('Numer gry')
    plt.ylabel('Wynik')
    #plt.show()
    plt.savefig("waz3_rozrz_stan")

    df = pd.read_csv('waz1_walls.csv')
    plt.figure(figsize=(14, 8))
    plt.plot(df.iloc[1:, 1:])
    plt.title("Wąż numer 1, ściany")
    plt.xlabel('Numer gry')
    plt.ylabel('Wynik')
    #plt.show()
    plt.savefig("waz1_walls")

    df = pd.read_csv('waz2_walls.csv')
    plt.figure(figsize=(14, 8))
    plt.plot(df.iloc[1:, 1:])
    plt.title("Wąż numer 2, ściany")
    plt.xlabel('Numer gry')
    plt.ylabel('Wynik')
    #plt.show()
    plt.savefig("waz2_walls")

    df = pd.read_csv('waz3_walls.csv')
    plt.figure(figsize=(14, 8))
    plt.plot(df.iloc[1:, 1:])
    plt.title("Wąż numer 3, ściany")
    plt.xlabel('Numer gry')
    plt.ylabel('Wynik')
    #plt.show()
    plt.savefig("waz3_walls")


if(__name__=="__main__"):
    plot1()

