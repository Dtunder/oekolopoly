"""
    Translation dictionaries for Ökolopoly GUI 2.0.

    Languages supported: "de" | "en"
"""


def dict_translate(language):
    """
    :param language: ``"de"|"en"``
    :return: translation dictionary for all GUI elements
    """
    if language == "de":
        dtl = {"HelpMode": "Hilfemodus",
               "ExitHelp": "Hilfe verlassen",
               "Exit": "Verlassen",
               "CloseGame": "Spiel beenden",
               "Forward": "Weiter \u2192",      # \u2192 is an arrow to the right
               "Back": "\u2190 Zurück",         # \u2190 is an arrow to the left
               "Page": "Seite: ",
               "SpecialCase": "Spezialfall",
               "APoints": "Aktionspunkte",
               'Reasoning': 'Begründung',
               'Prediction': 'Vorhersage (5J)',
               'HarmonyGrade': 'Harmonie-Note',
               'Penalty': 'Malus',
               "Points": "Punkte",
               "MoreInfoPoints": "Einfluss, Geld, Arbeit,\n"
                                 "Energie, Güter, Nahrung",
               "Redevelop": "Sanierung",
               "MoreInfoRedevelop": "Umweltschutz, Recycling,\n"
                                    "Biotechnik, sanfte Energie,\n"
                                    "Landschaftserhaltung,\n"
                                    "Humanisierung der\n"
                                    "Arbeitswelt",
               "Production": "Produktion",
               "MoreInfoProduction": "Industrie, Handwerk,\n"
                                     "Landwirtschaft,\n"
                                     "Dienstleistungen",
               "EnvirDamage": "Umweltbelastung",
               "MoreInfoEnvirDamage": "Abgase, Abwässer,\n"
                                      "Abwärme, Lärm, Raubbau,\n"
                                      "Landschaftszerstörung,\n"
                                      "Trennung natürlicher Kreisläufe,\n"
                                      "Verkehrschaos, Städtezerfall",
               "Enlightenment": "Aufklärung",
               "MoreInfoEnlighten": "Gesunde Lebensweise,\n"
                                    "Selbstverwirklichung, Schulen,\n"
                                    "Erwachsenenbildung, umweltbewusst,\n"
                                    "sinnvolle Freizeit,\n"
                                    "Bürgerinitiativen, Geburtenkontrolle",
               "QualityOfLife": "Lebensqualität",
               "MoreInfoQuality": "Gesundheit, Sicherheit,\n"
                                  "Sinnvolle Arbeit, Wohnqualität,\n"
                                  "Naherholung, Freizeitangebot",
               "ReproRate": "Vermehrungsrate",
               "MoreInfoReproRate": "Geburten, Sterbefälle,\n"
                                    "Unfälle, Zu- und Abwanderung",
               "Population": "Bevölkerung",
               "MoreInfoPopulation": "Bevölkerungszahl, Menschendichte,\n"
                                     "Altersaufbau, Arbeitskräfte,\n"
                                     "Sozialstruktur",
               "Politics": "Politik",
               "MoreInfoPolitics": "Weitsicht,\n"
                                   "Autorität,\n"
                                   "Beliebtheit,\n"
                                   "einsichtige\n"
                                   "Programme,\n"
                                   "Ent-\n"
                                   "scheidungs-\n"
                                   "gewalt",
               "MoreInfoCurrAction": "{:>72}".format("[S,   P,   A,   L,   V,   Sf]\n") +
                                     "mit S: Sanierung, P: Produktion, A: Aufklärung,\n"
                                     "L: Lebensqualität, V: Vermehrung, Sf: Spezialfall",
               "ClearActions": "Entferne verteilte Punkte",
               "ExecuteStep": "Zug ausführen",
               "Reset": "Neues Spiel",
               "BestMoveAI": "KI-Zug",
               "PreviewMode": "Vorschaumodus",
               "Help": "Hilfe?",
               "GameInstructions": "Spielanleitung",
               "GameHistory": "Spielhistorie",
               "ActionPointsLeft": "Aktionspunkte übrig: ",
               "DistributedPoints": "Verteilte Punkte: ",
               "Round": "Runde: ",
               "Balance": "Bilanz: ",
               "GameOver": "Game over",
               "FinalResult": "Endergebnis",
               "PointsGained": "Punkte erzielt",
               "RoundsSurvived": "Runden überlebt",
               "RoundNo": "Runde Nr.",
               "AttentionGameOver": "Vorsicht! `Game over` mit aktuellem Zug",
               }
        return dtl
    if language == "en":
        dtl = {"HelpMode": "Help Mode",
               "ExitHelp": "    Exit help",
               "Exit": "Exit game",
               "CloseGame": "Exit game",
               "Forward": "Forward \u2192",
               "Back": "\u2190 Back",
               "Page": "Page: ",
               "SpecialCase": "SpecialCase",
               "APoints": "Action points",
               'Reasoning': 'Reasoning',
               'Prediction': 'Prediction (5Y)',
               'HarmonyGrade': 'Harmony Grade',
               'Penalty': 'Penalty',
               "Points": "points",
               "MoreInfoPoints": "Influence, money, work,\n"
                                 "energy, goods, food",
               "Redevelop": "Redevelop",            # "Sanierung"
               "MoreInfoRedevelop": "Environmental protection,\n"
                                    "recycling, biotechnology,\n"
                                    "landscape conservation,\n"
                                    "humane working\n"
                                    "environment",
               "Production": "Production",
               "MoreInfoProduction": "Industry, crafts,\n"
                                     "agriculture,\n"
                                     "services",
               "EnvirDamage": "Environ. Damage",
               "MoreInfoEnvirDamage": "Waste gases, wastewater,\n"
                                      "waste heat, noise, exploitation,\n"
                                      "landscape destruction,\n"
                                      "disconnection of natural cycles,\n"
                                      "traffic chaos, urban decay",
               "Enlightenment": "Education",
               "MoreInfoEnlighten": "Healthy lifestyle, self-fulfilment, \n"
                                    "schools, adult education,\n"
                                    "environmentally conscious,\n"
                                    "meaningful leisure time,\n"
                                    "citizen initiatives, birth control",
               "QualityOfLife": "Quality of Life",
               "MoreInfoQuality": "Health, safety, meaningful work,\n"
                                  "quality of housing, local\n"
                                  "recreation, leisure activities",
               "ReproRate": "Reproduction Rate",
               "MoreInfoReproRate": "Births, deaths, accidents,\n"
                                    "immigration and emigration",
               "Population": "Population",
               "MoreInfoPopulation": "Population, population density,\n"
                                     "age structure, workforce,\n"
                                     "social structure",
               "Politics": "Politics",
               "MoreInfoPolitics": "Foresight,\n"
                                   "authority,\n"
                                   "popularity,\n"
                                   "reasonable\n"
                                   "programs,\n"
                                   "decision-\n"
                                   "making\n"
                                   "power",
               "MoreInfoCurrAction": "{:>72}".format("[R,  P,  E,  Q,  RR,  S]\n") +
                                     "with R: Redevelop, P: Production, E: Education,\n"
                                     "Q: Quality of Life, RR: ReproRate, S: Special Case",
               # "{:>66}".format is for right-aligned printing in a string field of size 66
               "ClearActions": "Remove distributed points",
               "ExecuteStep": "Execute step",
               "Reset": "New game",
               "BestMoveAI": "AI move",
               "PreviewMode": "Preview mode",
               "Help": "Help?",
               "GameInstructions": "Game instructions",
               "GameHistory": "Game history",
               "ActionPointsLeft": "Action points left: ",
               "DistributedPoints": "Distributed points: ",
               "Round": "Round: ",
               "Balance": "Balance: ",
               "GameOver": "Game over",
               "FinalResult": "Final result",
               "PointsGained": "points gained",
               "RoundsSurvived": "rounds survived",
               "RoundNo": "Round #",
               "AttentionGameOver": "Attention! `Game over` with actual move",
               }
        return dtl
    raise RuntimeError(f"[dict_translate] language = {language} is not supported!")


def env_translate(language):
    """
    :param language: ``"de"|"en"``
    :return: translation dictionary for all environment language elements
    """
    if language == "de":
        etl = {" too high. ": " zu hoch. ",
               " too low. ": " zu niedrig. ",
               " out of allowed range": " nicht im Bereich",
               "NumAPointsTooLow": "Zu wenig Aktionspunkte",
               "NumAPointsTooHigh": "Zu viele Aktionspunkte",
               "MaxNumRoundsReached": "Maximalzahl an Runden erreicht.",
               }
        return etl
    if language == "en":
        etl = {" too high. ": " too high. ",
               " too low. ": " too low. ",
               " out of allowed range": " out of allowed range",
               "NumAPointsTooLow": "Too few action points",
               "NumAPointsTooHigh": "Too many action points",
               "MaxNumRoundsReached": "Maximum number of rounds reached.",
               }
        return etl
    raise RuntimeError(f"[env_translate] language = {language} is not supported!")


def dict_help_screens(language):
    """
    :param language: ``"de"|"en"``
    :return: translation dictionary for all GUI help screens
    """
    if language == "de":
        htl = {
               "hs1.0": "Willkommen bei Ökolopoly!\n"
                        "Deine Aufgabe ist es, ein Land zu führen.\n\n"
                        "Du befindest dich gerade im Hilfemodus.\n"
                        "Hier wird das Wichtigste zum Spiel Ökolopoly\n"
                        "und seinen Features kurz erklärt.\n\n"
                        "Ziel des Spiels ist es, zuerst mindestens 10 Runden\n"
                        "zu überleben, da man erst nach 10 Runden Punkte \n"
                        "verdienen kann. Viele Punkte erhält man bei einem hohen \n"
                        "Wert im Bereich `Politik`. Gleichzeitig sollte dieser \n"
                        "Wert nach nicht zu vielen Runden (15-25) erreicht \n"
                        "werden.\n\n"
                        # "Zuerst werden 3 Spiele ohne neue Features gespielt.\n"
                        # "Verdeckte Features werden nach 3 zu Ende gespielten\n"
                        # "('Game over') Spielen je nach Gruppe freigeschaltet.\n\n"
                        # "Es wäre schön wenn mindestens 5 Spiele beendet werden.\n"
                        # f"Bereits beendete Spiele: {played_games}\n\n"
                        # "Bitte zum Beenden des Spiels den 'Spiel beenden'-Knopf\n"
                        # "verwenden, damit die Daten anonymisiert übermittelt\n"
                        # "werden können!\n\n"
                        "Viel Erfolg!",
               "hs2.0": "Es gibt 9 Bereiche auf die man\n"
                        "achten sollte: Sobald ein Bereich\n"
                        "seine obere oder untere Grenze \n"
                        "überschreitet, ist `Game over`!\n\n"
                        "5 der 9 Bereiche lassen sich\n"
                        "steuern. Die Zahl zwischen den \n"
                        "Buttons `+` und `-` kann man mit \n"
                        "eben diesen Buttons anpassen. \n"
                        "Die Zahl bestimmt die \n"
                        "Veränderungszahl für diesen Bereich.\n\n"
                        "Hinweis:\n"
                        "`Produktion` akzeptiert als\n"
                        "einziger Bereich auch negative\n"
                        "Veränderungszahlen!",
               "hs2.1": "Man erhält mehr Infos über einen Bereich,\n"
                        "wenn man mit der Maus darüber fährt.",
               "hs2.2": "Immer rechts neben dem Balken steht\n"
                        "der aktuelle Wert des Bereiches.",
               "hs2.3": "Zu Beginn jeder Runde wird eine bestimme Anzahl an\n"
                        "Aktionspunkten bereitgestellt. Diese kann man beliebig auf\n"
                        "die 5 Bereiche verteilen.\n\n"
                        "Mit Button `Entferne verteilte Punkte` werden alle in\n"
                        "dieser Runde verteilten Aktionspunkte zurückgesetzt.",
               "hs3.0": "Mit Button `Neues Spiel` kann\n"
                        "ein neues Spiel gestartet werden.\n"
                        "Der bisherige Fortschritt geht dabei\n"
                        "aber verloren!\n\n" 
                        "Die Runde zeigt an in welchem Jahr\n"
                        "man sich befindet. Man beginnt bei Runde 0.",
               "hs4.0": "Mit `Zug ausführen` schließt man die aktuelle\n"
                        "Runde ab und die verteilten Aktionspunkte werden angewendet.\n"
                        "Einige der Bereiche haben Auswirkungen auf andere und diese\n"
                        "Auswirkungen werden mit Abschluss der Runde zusätzlich eintreten!",
               "hs5.0": "Der Spezialfall tritt ein, wenn `Aufklärung` einen Wert von 20 oder höher erreicht hat.\n"
                        "Nun kann man die Vermehrungsrate an einem weiteren Ort anpassen. Und vor allem:\n"
                        "Man kann nun auch negative Änderungen für die Vermehrungsrate eintragen!\n"
                        "Der weitere Vorteil:\n"
                        "Es werden keine Aktionspunkte benötigt! \n"
                        "Der Nachteil:\n"
                        "Der Spezialfall erlaubt nur Änderungen von -5 bis +5. Wenn `Aufklärung` zwischen\n"
                        "20 und 22 liegt, haben sogar nur Werte von -3 bis +3 einen Effekt. Siehe Grafik.",
               "hs6.0": "Das ist die Konsole. Hier wird nach dem Ende eines Spiels Feedback gegeben:\n"
                        "Warum das Spiel zu Ende ist und wie viele Punkte man erzielt hat.",
               "hs7.0": "Mit Button `Spielanleitung` öffnet sich die originale\n"
                        "Spielanleitung des Spiels Ökolopoly.\n\n"
                        "Mit Button `Spielhistorie` kann man sich die bereits gespielten Spiele anschauen.\n\n"
                        "Button `Hilfe?` führt zu diesem Hilfemodus.\n\n\n\n"
                        "Mit den Tasten W-A-S-D oder den Pfeiltasten auf der Tastatur\n"
                        "kann man sich frei auf dem Spielbrett bewegen und mit dem\n"
                        "Mausrad hinein- und herauszoomen.\n"
                        "Beim maximalen Herauszoomen zentriert sich das Spielbrett automatisch.\n"
                        "Mit der Tabulatortaste lassen sich alle Spielelemente ein- und ausblenden.\n"
                        "Mit Taste 1 kann man zwischen verschiedenen Hintergrundbildern wechseln.\n\n"
                        "Achtung:\n"
                        "Die Spielelemente lassen sich nur im maximal herausgezoomten Zustand bedienen.\n\n"
                        "Wichtige Anmerkung:\n"
                        "Um das Spiel Ökolopoly zu meistern, lohnt es sich, alle Diagramme auf den letzten Seiten\n"
                        "der Spielanleitung zu studieren.",
               "hs8.0": "Die Feature-Balken geben Auskunft über die\n"
                        "minimalen und maximalen Werte, die jeder Bereich\n"
                        "annehmen kann.\n\n"
                        "Abhängig vom momentanen Wert ist die Farbe des Balken\n"
                        "grün (in der Mitte) und wird zu den beiden\n"
                        "Rändern hin rot.",
               "hs9.0": "Das ist der Button `Vorschaumodus`\n"
                        "Damit schaltet man den Vorschaumodus\n"
                        "an und aus.",
               "hs9.1": "Der Vorschaumodus zeigt an, wie sich die Bereiche mit den\n"
                        "aktuell verteilten Aktionspunkten in der nächsten Runde \n"
                        "verändern würden.\n"
                        "Ein schwarzer Strich zeigt den neuen Wert des Balken an.\n"
                        "Außerdem wird mit Hilfe einer Zahl über beziehungsweise\n"
                        "- bei negativen Zahlen - unter dem aktuellen Wert die \n"
                        "Abnahme/Zunahme des Bereiches angezeigt.\n\n"
                        "Hinweis:"
                        "Sind diese grün, überlebt man die Runde. Sind diese jedoch\n"
                        "rot, so ist `Game over`, falls man den Zug ausführt.\n"
                        "Die Konsole gibt Auskunft dazu!",
               "hs9.2": "Das ist die Konsole des Vorschaumodus. Diese gibt Auskunft über den Grund\n"
                        "für `Game over`, falls man den aktuellen Zug ausführen würde.",
               "hs10.0": "Der Button `KI-Zug` verteilt die verfügbaren Aktionspunkte\n"
                         "bestmöglich auf die Bereiche.\n"
                         "Dahinter steckt ein KI-Agent, der auf diese Aufgabe trainiert wurde.\n\n"
                         "Wenn bin/config.txt 'predict_usages' < 6 hat, dann ist nur diese Anzahl an KI-Zügen erlaubt.\n\n",
               "hs10.1": "Achtung:\n"
                         "Die Nutzung der KI ist auf",
               "hs10.2": "-mal pro Spiel begrenzt!",
        }
        return htl
    if language == "en":
        htl = {
               "hs1.0": "Welcome to Ökolopoly!\n"
                        "Your task is to lead a country.\n\n"
                        "You are currently in help mode.\n"
                        "The help mode explains the most \n"
                        "important aspects of the game Ökolopoly.\n\n"
                        "The aim of the game is at first to survive 10 rounds,\n"
                        "as you can only earn points after 10 rounds. \n"
                        "Thereafter you want to gain as much points as possible.\n"
                        "You gain many points for a high value in sector `Politics`. \n"
                        "At the same time, this value should be achieved after \n"
                        "not too many rounds (15-25).\n\n"
                        # "Zuerst werden 3 Spiele ohne neue Features gespielt.\n"
                        # "Verdeckte Features werden nach 3 zu Ende gespielten\n"
                        # "('Game over') Spielen je nach Gruppe freigeschaltet.\n\n"
                        # "Es wäre schön wenn mindestens 5 Spiele beendet werden.\n"
                        # f"Bereits beendete Spiele: {played_games}\n\n"
                        # "Bitte zum Beenden des Spiels den 'Spiel beenden'-Knopf\n"
                        # "verwenden, damit die Daten anonymisiert übermittelt\n"
                        # "werden können!\n\n"
                        "Good luck!",
               "hs2.0": "There are 9 sectors to watch out for:\n"
                        "As soon as an sector exceeds its upper \n"
                        "or lower limit, it's game over!\n\n"
                        "5 of the 9 sectors can be controlled.\n"
                        "The number between the `+` and `-` \n"
                        "buttons can be adjusted using these \n"
                        "buttons. The number determines the \n"
                        "change value for this sector.\n\n"
                        "Note:\n"
                        "`Production` is the only sector that \n"
                        "accepts negative change values!",
               "hs2.1": "You can get more information about an \n"
                        "sector by moving your mouse over it.",
               "hs2.2": "To the right of the bar is\n"
                        "the current value of the range.",
               "hs2.3": "At the beginning of each round, a certain number of\n"
                        "action points is provided. Those points can distributed\n"
                        "as desired among the 5 sectors.\n\n"
                        "The `Remove distributed points` button resets\n"
                        "all action points distributed in this round.",
               "hs3.0": "With the ‘New Game’ button,\n"
                        "a new game can be started.\n"
                        "However, any progress made \n"
                        "so far will be lost!\n\n"
                        "The round indicates which year you are in.\n"
                        "You start at round 0.",
               "hs4.0": "With `Execute Move` you complete the current\n"
                        "round and the distributed action points are applied.\n"
                        "Some of the sectors have effects on others, and these\n"
                        "effects will also occur at the end of the round!",
               "hs5.0": "The special case occurs when `Eductation` has reached a value of 20 or higher.\n"
                        "Now you can adjust `Reproduction rate` in another location. And most importantly:\n"
                        "You can now also enter negative changes for the reproduction rate!\n"
                        "The additional advantage:\n"
                        "No action points are needed! \n"
                        "“The disadvantage:\n"
                        "The special case only allows changes from -5 to +5. If `Education` is between\n"
                        "20 und 22, only values from -3 to +3 have an effect. See diagram.",
               "hs6.0": "This is the console. Feedback is provided here after the end of a game:\n"
                        "Why the game is over and how many points you have scored.",
               "hs7.0": "Button `Game instructions` opens the original game instructions of the\n"
                        "game Ökolopoly (sorry, in German only).\n\n"
                        "With button `Game history` you can view a text log of the already played episodes.\n\n"
                        "Button `Help?` leads to this Help Mode.\n\n\n\n"
                        "Using the W-A-S-D keys or the arrow keys on the keyboard, you can move\n"
                        "freely around the game board and use the mouse wheel to zoom in and out.\n"
                        "When zoomed out to the maximum, the game board automatically centers itself.\n"
                        "Use the Tab key to show and hide all game elements.\n"
                        "Key 1 toggles between different background images.\n\n"
                        "Please note:\n"
                        "The game elements can only be operated when zoomed out to the maximum.\n\n"
                        "Important extra note:\n"
                        "To master the game Ökolopoly, it is worth studying all the diagrams on the last\n"
                        "pages of the game instructions. (The names at the axes are the sector names in German. The \n"
                        "German-English translation of the sectors is visible by using the Tab key repeatedly.)",
               "hs8.0": "The feature bars provide information about the minimum \n"
                        "and maximum values that each sector can assume.\n\n"
                        "Depending on the current value, the color of the bar\n"
                        "is green (value in the middle) and turns red, \n"
                        "if the value moves towards upper or lower limit.",
               "hs9.0": "This is the button `Preview mode`\n"
                        "It turns preview mode on and off.",
               "hs9.1": "The preview mode shows how the sectors with the\n"
                        "currently distributed action points would change in the \n"
                        "next round.\n"
                        "A black line indicates the new value in the bar.\n"
                        "In addition, a number above or - in the case of negative numbers - \n"
                        "below the current value indicates the decrease/increase in the sector.\n\n"
                        "Note:\n"
                        "If these numbers are green, you survive the round. However, if \n"
                        "they are red, it's ‘Game over’ if you execute the move.\n"
                        "The preview mode console provides information about this!",
               "hs9.2": "This is the preview mode console. It provides information about the reason\n"
                        "for ‘Game over’ if you were to execute the planned move.",
               "hs10.0": "The ‘AI move’ button distributes the available action points\n"
                        "as best as possible across the sectors.\n"
                        "Behind this is an AI agent that has been trained for this task.\n\n"
                        "If bin/config.txt has `predict_usages` < 6, only this number of AI moves is allowed.\n\n",
               "hs10.1": "Attention:\n"
                         "Use of AI is limited to ",
               "hs10.2": " times per game!",
               }
        return htl
    raise RuntimeError(f"[dict_help_screens] language = {language} is not supported!")
