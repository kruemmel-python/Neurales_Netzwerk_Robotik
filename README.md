# Neuronales Netzwerk für Roboter-Navigation und Hinderniserkennung

Dieses Projekt umfasst ein **biologisch inspiriertes neuronales Netzwerk**, das auf **Hebb'schem Lernen** basiert. Das Modell wurde entwickelt, um **Lernprozesse** zu simulieren, bei denen **Knoten in einem Netzwerk** auf Basis von **lokalen Interaktionen und räumlichen Beziehungen** zusammenarbeiten. Das Netzwerk wird verwendet, um **Sensordaten eines Roboters** zu analysieren, Hindernisse zu erkennen und eine Entscheidung über die Navigation zu treffen.

## Besonderheiten des Modells

- **Biologisch inspiriertes Hebb’sches Lernen**:
  Anstatt des traditionellen Backpropagation-Ansatzes wird hier **Hebb’sches Lernen** verwendet. Dies bedeutet, dass Verbindungen zwischen Knoten basierend auf der gleichzeitigen Aktivierung dieser Knoten verstärkt werden – ein Lernprinzip, das an biologische Gehirnprozesse angelehnt ist. Dieses **dezentrale Lernen** ermöglicht eine natürlichere und flexiblere Anpassung des Modells an neue Daten.

- **Erweiterte Netzwerktopologie**:
  Das Netzwerk besteht aus **Knoten**, die miteinander verbunden sind. Diese Verbindungen werden dynamisch auf Grundlage der **räumlichen Nähe** der Knoten und eines **harmonischen Radius** geschaffen. Das Modell nutzt also **räumliche Beziehungen** zwischen Knoten und erzeugt so eine flexible, nicht-lineare Netzwerktopologie, die auf lokale und globale Interaktionen reagiert.

- **Kollaborative Signalweitergabe**:
  Das Modell ist darauf ausgelegt, **Signale zwischen den Knoten zu propagieren**, wobei jede Verbindung auf der lokalen Aktivierung der Knoten basiert. Im Gegensatz zu traditionellen neuronalen Netzwerken, bei denen die Eingabedaten durch vordefinierte Schichten fließen, ermöglicht das Modell eine **kollaborative Signalweitergabe**, bei der alle Knoten zusammenarbeiten, um zu einer Entscheidung zu kommen.

- **Selbstverstärkende Lernprozesse**:
  Durch Hebb’sches Lernen verstärken sich **Verbindungen**, wenn zwei Knoten gleichzeitig aktiviert werden, was zu einer **selbstorganisierenden Struktur** führt. Dieser Lernmechanismus sorgt dafür, dass das Netzwerk mit der Zeit immer besser in der Lage ist, Muster zu erkennen und Entscheidungen zu treffen.

## Anwendung

### 1. **Roboter-Sensordaten und Navigation**
   Das Modell ist besonders geeignet für Anwendungen, bei denen es darum geht, **Hindernisse zu erkennen** und Entscheidungen zur **Navigation** in einer 2D-Welt zu treffen. Ein Beispiel ist die **Simulierte Roboter-Navigation**, bei der das Modell LIDAR-ähnliche Sensordaten verwendet, um Hindernisse zu erkennen und die Bewegungsrichtung zu bestimmen.

### 2. **Simulierte LIDAR-Scans und Hinderniserkennung**
   Durch die Simulation von **LIDAR-Scans** wird dem Modell ermöglicht, die **Position und Form von Hindernissen** in einer Umgebung zu erkennen. Diese Daten können dann verwendet werden, um auf die Umwelt zu reagieren und Entscheidungen zu treffen.

### 3. **Potenzial für echte Roboteranwendungen**
   Das Modell könnte theoretisch in Roboteranwendungen eingesetzt werden, bei denen **räumliche Beziehungen und kollaborative Signalverarbeitung** erforderlich sind. Insbesondere für Aufgaben wie **Simultane Lokalisierung und Kartierung (SLAM)** und **Objekterkennung in Echtzeit** könnte dieses Modell nützlich sein.

## Unterschiede zu traditionellen neuronalen Netzwerken

- **Traditionelle neuronale Netzwerke** wie **Convolutional Neural Networks (CNNs)** oder **Recurrent Neural Networks (RNNs)** verwenden vorgegebene Schichten und mathematische Optimierungstechniken wie **Backpropagation** zur Anpassung der Gewichte. Diese Modelle sind stark auf zentrale Fehlerbackpropagation angewiesen und arbeiten in festen Architekturen.

- **Hebb’sches Lernen**, wie es in diesem Projekt verwendet wird, basiert auf der **gleichzeitigen Aktivierung von Knoten**, wodurch das Netzwerk auf natürliche Weise Verbindungen verstärkt und die Lernprozesse nicht zentralisiert, sondern dezentralisiert und **selbstorganisierend** sind.

- Das Modell unterscheidet sich durch seine **dynamische Netzwerkstruktur**, die Knoten **spontan miteinander verbindet**, je nach räumlicher Nähe und Interaktion. Dies führt zu einem **adaptiven, flexiblen Netzwerk**, das in der Lage ist, mit komplexeren und variablen Datensätzen besser umzugehen.

## Installation

Stelle sicher, dass du **Python 3.7+** installiert hast. Alle Abhängigkeiten können mit folgendem Befehl installiert werden:

```bash
pip install -r requirements.txt
```

Die `requirements.txt` enthält die folgenden wichtigen Pakete:

- **matplotlib**: Für die Visualisierung des Netzwerks und der Simulationsergebnisse.
- **numpy**: Für mathematische Operationen und die Arbeit mit Vektoren und Matrizen.
- **pandas**: Zum Verarbeiten und Erstellen von Datensätzen.
- **tqdm**: Für die Fortschrittsanzeige bei der Modelltrainierung.
- **scikit-learn**: Zum Aufteilen der Daten in Trainings- und Testsets.

## Verwendung

### 1. **Trainiere das Netzwerk**

   Verwende die folgenden Schritte, um das Modell zu trainieren. Der Datensatz wird automatisch generiert, aber du kannst auch deine eigenen Daten verwenden.

```bash
python recognition_model.py
```

### 2. **Simuliere die Navigation**

   Das Modell nutzt **LIDAR-ähnliche Sensordaten**, um Hindernisse in der Umgebung zu erkennen und die Navigation des Roboters zu simulieren. Diese Daten können dann genutzt werden, um Entscheidungen zur Bewegung und Interaktion mit der Umgebung zu treffen.

### 3. **Testen des Netzwerks**

   Nach dem Training kannst du das Modell testen, um zu sehen, wie gut es bei der **Hinderniserkennung** und der **Navigation** funktioniert. Dabei werden die **LIDAR-Daten** und **Aktivierungen des Netzwerks** visualisiert.

## Zusammengefasst

Dieses Projekt stellt ein **selbstorganisierendes neuronales Netzwerk** dar, das **biologisch inspiriertes Hebb’sches Lernen** verwendet, um Hindernisse zu erkennen und auf eine dynamische Umgebung zu reagieren. Die Kombination von **Hebb’schem Lernen** und einer **dynamischen Netzwerkstruktur** hebt es von traditionellen neuronalen Netzwerken ab und ermöglicht eine anpassungsfähige und selbstverstärkende Lernweise, die besonders für Anwendungen in der **Roboternavigation** und **Sensorfusion** geeignet ist.
