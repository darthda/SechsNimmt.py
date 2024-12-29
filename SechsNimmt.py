import random
import matplotlib.pyplot as plt
import numpy as np

class Karte:
    def __init__(self, nummer):
        self.nummer = nummer
        self.strafpunkte = self.berechne_strafpunkte()

    def berechne_strafpunkte(self):
        if self.nummer % 10 == 0:
            return 3
        elif self.nummer % 5 == 0:
            return 2
        elif self.nummer % 11 == 0:
            return 5
        else:
            return 1

    def __repr__(self):
        return f"Karte({self.nummer}, Strafpunkte: {self.strafpunkte})"

class Deck:
    def __init__(self):
        self.karten = [Karte(nummer) for nummer in range(1, 105)]
        random.shuffle(self.karten)

    def ziehe_karte(self):
        return self.karten.pop() if self.karten else None

class Spieler:
    def __init__(self, name, strategie="random", genetische_strategie=None):
        self.name = name
        self.hand = []
        self.strafpunkte = 0
        self.strategie = strategie
        self.genetische_strategie = genetische_strategie

    def ziehe_karten(self, deck, anzahl):
        for _ in range(anzahl):
            karte = deck.ziehe_karte()
            if karte:
                self.hand.append(karte)

    def spiele_karte(self, reihen):
        if self.strategie == "random":
            return self.hand.pop(random.randint(0, len(self.hand) - 1))
        elif self.strategie == "regelbasiert":
            best_card = None
            for karte in self.hand:
                passende_reihe = None
                kleinste_diff = float("inf")

                for reihe in reihen:
                    oberste = reihe.oberste_karte()
                    if oberste and oberste.nummer < karte.nummer:
                        diff = karte.nummer - oberste.nummer
                        if diff < kleinste_diff:
                            passende_reihe = reihe
                            kleinste_diff = diff

                if passende_reihe and passende_reihe.kann_legen(karte):
                    return self.hand.pop(self.hand.index(karte))

            return self.hand.pop(self.hand.index(min(self.hand, key=lambda k: k.nummer)))
        elif self.strategie == "genetisch":
            scores = []
            for karte in self.hand:
                score = 0
                for reihe in reihen:
                    oberste = reihe.oberste_karte()
                    if oberste and oberste.nummer < karte.nummer:
                        score += self.genetische_strategie.get((oberste.nummer, karte.nummer), 1)
                scores.append(score)

            beste_karte_index = np.argmax(scores)
            return self.hand.pop(beste_karte_index)

    def waehle_reihe(self, reihen):
        return min(reihen, key=lambda reihe: reihe.gesamtschaden())

    def __repr__(self):
        return f"Spieler({self.name}, Strafpunkte: {self.strafpunkte})"

class Reihe:
    def __init__(self):
        self.karten = []

    def kann_legen(self, karte):
        return len(self.karten) < 5

    def lege_karte(self, karte):
        self.karten.append(karte)

    def gesamtschaden(self):
        return sum(karte.strafpunkte for karte in self.karten)

    def oberste_karte(self):
        return self.karten[-1] if self.karten else None

    def leere_reihe(self, neue_karte):
        alte_karten = self.karten
        self.karten = [neue_karte]
        return alte_karten

    def __repr__(self):
        return f"Reihe({self.karten})"

class GenetischerAlgorithmus:
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [self.erzeuge_individuum() for _ in range(self.population_size)]

    def erzeuge_individuum(self):
        return {(i, j): random.uniform(0, 1) for i in range(1, 105) for j in range(1, 105) if i < j}

    def bewertung(self, individuum, spieler_name):
        spiel = HornochseSpiel(["Bot1", "Bot2", spieler_name])
        spiel.spieler[2].genetische_strategie = individuum
        spiel.spieler[2].strategie = "genetisch"
        spiel.starte_partie()
        return -spiel.spieler[2].strafpunkte

    def auswahl(self):
        fitness = [self.bewertung(individuum, "Bot3") for individuum in self.population]
        fitness_summe = sum(fitness)
        if fitness_summe == 0:
            return random.choices(self.population, k=2)
        wahrscheinlichkeiten = [f / fitness_summe for f in fitness]
        return random.choices(self.population, weights=wahrscheinlichkeiten, k=2)

    def crossover(self, eltern):
        kind = {}
        for key in eltern[0]:
            kind[key] = random.choice([eltern[0][key], eltern[1][key]])
        return kind

    def mutation(self, individuum):
        for key in individuum:
            if random.random() < self.mutation_rate:
                individuum[key] = random.uniform(0, 1)

    def neue_generation(self):
        neue_population = []
        for _ in range(self.population_size):
            eltern = self.auswahl()
            kind = self.crossover(eltern)
            self.mutation(kind)
            neue_population.append(kind)
        self.population = neue_population

class HornochseSpiel:
    def __init__(self, spieler_namen):
        self.deck = Deck()
        self.spieler = [
            Spieler(name, strategie="random" if name == "Bot1" else "regelbasiert" if name == "Bot2" else "genetisch")
            for name in spieler_namen
        ]
        self.reihen = [Reihe() for _ in range(4)]
        self.statistiken = {spieler.name: [] for spieler in self.spieler}
        self.wins = {spieler.name: 0 for spieler in self.spieler}
        self.bot3_punkte = []
        self.siege_pro_episode = {spieler.name: [] for spieler in self.spieler}

    def starte_spiel(self):
        for spieler in self.spieler:
            spieler.ziehe_karten(self.deck, 10)
        for reihe in self.reihen:
            reihe.lege_karte(self.deck.ziehe_karte())

    def spiele_karte(self, spieler, karte):
        passende_reihe = None
        kleinste_diff = float('inf')

        for reihe in self.reihen:
            oberste = reihe.oberste_karte()
            if oberste and oberste.nummer < karte.nummer:
                diff = karte.nummer - oberste.nummer
                if diff < kleinste_diff:
                    passende_reihe = reihe
                    kleinste_diff = diff

        if passende_reihe and passende_reihe.kann_legen(karte):
            passende_reihe.lege_karte(karte)
        else:
            self.wahle_und_nimm_reihe(spieler, karte)

    def wahle_und_nimm_reihe(self, spieler, karte):
        reihe = spieler.waehle_reihe(self.reihen)
        spieler.strafpunkte += reihe.gesamtschaden()
        reihe.leere_reihe(karte)

    def runde_spielen(self):
        for spieler in self.spieler:
            if spieler.hand:
                karte = spieler.spiele_karte(self.reihen)
                self.spiele_karte(spieler, karte)

    def spiel_beenden(self):
        return all(not spieler.hand for spieler in self.spieler)

    def starte_partie(self):
        self.starte_spiel()
        while not self.spiel_beenden():
            self.runde_spielen()

    def trainiere(self, episoden, genetischer_algorithmus):
        for episode in range(episoden):
            print(f"Starte Episode {episode + 1}")
            genetischer_algorithmus.neue_generation()
            self.spieler[2].genetische_strategie = genetischer_algorithmus.population[0]
            self.reset_spiel()
            self.starte_partie()
            self.auswerten_metriken()

    def reset_spiel(self):
        self.deck = Deck()
        for spieler in self.spieler:
            spieler.hand = []
            spieler.strafpunkte = 0
        self.reihen = [Reihe() for _ in range(4)]

    def auswerten_metriken(self):
        min_punkte = min(spieler.strafpunkte for spieler in self.spieler)
        gewinner = [spieler for spieler in self.spieler if spieler.strafpunkte == min_punkte]

        for spieler in self.spieler:
            self.statistiken[spieler.name].append(spieler.strafpunkte)

        for spieler in self.spieler:
            if spieler.name == "Bot3":
                self.bot3_punkte.append(spieler.strafpunkte)

        for spieler in self.spieler:
            self.siege_pro_episode[spieler.name].append(1 if spieler in gewinner else 0)

        for sieger in gewinner:
            self.wins[sieger.name] += 1 / len(gewinner)

        gewinner_namen = ", ".join(f"{sieger.name} ({sieger.strafpunkte} Punkte)" for sieger in gewinner)
        print(f"Gewinner der Episode: {gewinner_namen}")
        print(f"Aktueller Stand Wins: {self.wins}")

    def plot_metrics(self):
        plt.figure(figsize=(16, 6))

        # Plot der Gewinnwahrscheinlichkeiten
        plt.subplot(1, 3, 1)
        gesamtspiele = sum(self.wins.values())
        for spieler, siege in self.wins.items():
            plt.bar(spieler, siege / gesamtspiele if gesamtspiele > 0 else 0)
        plt.title("Gewinnwahrscheinlichkeiten")
        plt.ylabel("Gewinnquote")
        plt.xlabel("Spieler")

        # Plot der gesammelten Strafpunkte
        plt.subplot(1, 3, 2)
        for spieler, punkte in self.statistiken.items():
            plt.plot(range(1, len(punkte) + 1), punkte, label=spieler)
        plt.title("Strafpunkte über Episoden")
        plt.ylabel("Strafpunkte")
        plt.xlabel("Episode")
        plt.legend()

        # Plot der Strafpunkte von Bot3 separat
        plt.subplot(1, 3, 3)
        plt.plot(range(1, len(self.bot3_punkte) + 1), self.bot3_punkte, label="Bot3 Strafpunkte", color="red")
        plt.title("Bot3 Strafpunkte")
        plt.ylabel("Strafpunkte")
        plt.xlabel("Episode")
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Plot der kumulativen Siege pro Episode
        plt.figure(figsize=(8, 4))
        for spieler, siege in self.siege_pro_episode.items():
            kumulative_siege = np.cumsum(siege)
            plt.plot(range(1, len(kumulative_siege) + 1), kumulative_siege, label=spieler)
        plt.title("Kumulative Siege über Episoden")
        plt.ylabel("Kumulative Siege")
        plt.xlabel("Episode")
        plt.legend()
        plt.show()

spiel = HornochseSpiel(["Bot1", "Bot2", "Bot3"])
genetischer_algorithmus = GenetischerAlgorithmus(population_size=10, mutation_rate=0.05)
spiel.trainiere(episoden=100, genetischer_algorithmus=genetischer_algorithmus)
spiel.plot_metrics()
