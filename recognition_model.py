import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- Hilfsfunktionen ---
def distance(pos1, pos2):
    """Berechnet die Distanz zwischen zwei Punkten."""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def sigmoid(x):
    """Sigmoid-Aktivierungsfunktion."""
    # Überprüfen, ob x ein Array ist, und den Mittelwert berechnen, falls ja
    if isinstance(x, np.ndarray):
        x = np.mean(x)  # Wenn es ein Array ist, verwenden wir den Mittelwert
    return 1 / (1 + np.exp(-x))

# --- Klassen für Netzwerkstruktur ---
class Connection:
    def __init__(self, target_node, weight=None):
        self.target_node = target_node
        self.weight = weight if weight is not None else random.uniform(0.1, 1.0)

class Node:
    def __init__(self, position):
        self.position = position
        self.connections = []
        self.activation = 0.0  # Aktivierungswert des Knotens

    def add_connection(self, target_node, weight=None):
        self.connections.append(Connection(target_node, weight))

# --- Netzwerkinitialisierung ---
def initialize_network(num_nodes, V_max, R_harmonie):
    """Erstellt ein Netzwerk mit Knoten und Verbindungen."""
    nodes = [Node((0, 0))]  # Start mit zentralem Knoten

    for _ in range(1, num_nodes):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        new_node = Node((x, y))

        # Verbindungen innerhalb des harmonischen Radius
        for node in nodes:
            if distance(node.position, new_node.position) <= R_harmonie and len(node.connections) < V_max:
                node.add_connection(new_node)
                new_node.add_connection(node)

        nodes.append(new_node)

    return nodes

# --- Verbindung isolierter Knoten ---
def connect_isolated_nodes(nodes):
    """Verbindet isolierte Knoten mit dem Hauptnetzwerk."""
    for node in nodes:
        if len(node.connections) == 0:
            nearest_node = min(
                nodes,
                key=lambda n: distance(node.position, n.position)
            )
            node.add_connection(nearest_node)
            nearest_node.add_connection(node)

# --- Signalweitergabe ---
def propagate_signal(node, input_signal, visited=None):
    """Propagiert ein Signal durch das Netzwerk, ohne Zyklen zu erzeugen."""
    if visited is None:
        visited = set()
    if node in visited:
        return
    visited.add(node)

    # Sicherstellen, dass input_signal ein Skalar ist
    if isinstance(input_signal, np.ndarray):
        input_signal = np.mean(input_signal)  # Berechne den Mittelwert, wenn es ein Array ist

    node.activation = sigmoid(input_signal)
    for connection in node.connections:
        propagated_signal = node.activation * connection.weight
        propagate_signal(connection.target_node, propagated_signal, visited)

# --- Hebb'sches Lernen ---
def hebbian_learning(node, learning_rate=0.1):
    """Hebb'sches Lernen zur Anpassung der Verbindungsgewichte."""
    for connection in node.connections:
        connection.weight += learning_rate * node.activation * connection.target_node.activation

# --- Training ---
def train_network(nodes, data, epochs, learning_rate):
    """Trainiert das Netzwerk mit Eingabedaten."""
    weights_history = []

    for epoch in range(epochs):
        print(f"Start von Epoche {epoch + 1}/{epochs}...")
        epoch_progress = tqdm(data, desc=f"Training Epoche {epoch + 1}")
        for input_signal in epoch_progress:
            propagate_signal(nodes[0], input_signal)
            for node in nodes:
                hebbian_learning(node, learning_rate)

        weights = [conn.weight for node in nodes for conn in node.connections]
        weights_history.append(weights)

        print(f"Epoche {epoch + 1} abgeschlossen.")

    return weights_history

# --- Visualisierung ---
def visualize_network(nodes):
    """Visualisiert das Netzwerk mit Aktivierungen und Verbindungen."""
    plt.figure(figsize=(10, 10))
    for node in nodes:
        for connection in node.connections:
            x1, y1 = node.position
            x2, y2 = connection.target_node.position
            plt.plot([x1, x2], [y1, y2], color="gray", alpha=0.5)

    for node in nodes:
        x, y = node.position
        size = node.activation * 100 + 10
        color = (node.activation, 0.5, 1 - node.activation)
        plt.scatter(x, y, s=size, color=color, alpha=0.8)

    plt.axis("equal")
    plt.title("Netzwerk nach dem Training")
    plt.show()

def visualize_weights_over_time(weights_history):
    plt.figure(figsize=(10, 6))
    for epoch, weights in enumerate(weights_history):
        plt.plot(weights, label=f'Epoche {epoch + 1}')
    plt.xlabel("Verbindungen")
    plt.ylabel("Gewicht")
    plt.title("Entwicklung der Verbindungsgewichte über die Zeit")
    plt.legend()
    plt.show()

# --- Anwendung auf neue Daten ---
def test_network(nodes, test_data):
    correct_predictions = 0
    for input_signal in test_data:
        propagate_signal(nodes[0], input_signal)
        if nodes[0].activation > 0.5:  # Beispiel für eine Klassifikation
            correct_predictions += 1
    print(f"Genauigkeit: {correct_predictions / len(test_data):.2f}")

# --- LIDAR-ähnliche Sensordaten für Robotersimulation ---
def generate_obstacles(num_obstacles, world_size=10):
    """Erstellt zufällige Hindernisse in der 2D-Welt (Kreise und Rechtecke)."""
    obstacles = []
    for _ in range(num_obstacles):
        shape = random.choice(['circle', 'rectangle'])
        x = random.uniform(0, world_size)
        y = random.uniform(0, world_size)
        size = random.uniform(0.5, 1.5)
        obstacles.append((shape, x, y, size))
    return obstacles

def lidar_scan(robot_position, obstacles, scan_range=5, num_beams=360):
    """Simuliert die LIDAR-Daten eines Roboters in einer 2D-Welt."""
    lidar_data = []
    for angle in np.linspace(0, 2 * np.pi, num_beams):
        # Berechne das Ende des LIDAR-Strahls
        end_x = robot_position[0] + scan_range * np.cos(angle)
        end_y = robot_position[1] + scan_range * np.sin(angle)

        min_distance = scan_range
        for shape, x, y, size in obstacles:
            if shape == 'circle':
                obstacle_distance = np.sqrt((end_x - x) ** 2 + (end_y - y) ** 2)
                if obstacle_distance <= size:
                    min_distance = min(min_distance, obstacle_distance)
            elif shape == 'rectangle':
                if x - size / 2 <= end_x <= x + size / 2 and y - size / 2 <= end_y <= y + size / 2:
                    min_distance = 0

        lidar_data.append((angle, min_distance))

    return lidar_data

# --- Simulation der Navigation ---
def robot_navigation():
    # Initialisiere den Roboter
    robot_position = [5, 5]  # Startposition des Roboters in der Mitte der Welt
    obstacles = generate_obstacles(5)  # Erstelle 5 zufällige Hindernisse

    # Führe eine LIDAR-Scan durch
    lidar_data = lidar_scan(robot_position, obstacles)

    # Visualisiere die Welt und die LIDAR-Daten
    plot_world(robot_position, obstacles)
    plot_lidar_data(robot_position, lidar_data)

    # Gebe LIDAR-Daten aus
    print(f"LIDAR-Daten (Winkel und Distanz): {lidar_data}")

def plot_world(robot_position, obstacles):
    """Zeichnet die Welt und den Roboter."""
    plt.figure(figsize=(8, 8))
    # Zeichne Hindernisse
    for shape, x, y, size in obstacles:
        if shape == 'circle':
            circle = plt.Circle((x, y), size, color='red', alpha=0.5)
            plt.gca().add_artist(circle)
        elif shape == 'rectangle':
            rect = plt.Rectangle((x - size / 2, y - size / 2), size, size, color='blue', alpha=0.5)
            plt.gca().add_artist(rect)

    # Zeichne den Roboter
    plt.scatter(robot_position[0], robot_position[1], color='green', s=100)

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def plot_lidar_data(robot_position, lidar_data):
    """Zeichnet die LIDAR-Daten."""
    plt.figure(figsize=(8, 8))
    # Zeichne den Roboter
    plt.scatter(robot_position[0], robot_position[1], color='green', s=100)

    # Zeichne die LIDAR-Strahlen
    for angle, distance in lidar_data:
        end_x = robot_position[0] + distance * np.cos(angle)
        end_y = robot_position[1] + distance * np.sin(angle)
        plt.plot([robot_position[0], end_x], [robot_position[1], end_y], color='red', alpha=0.5)

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# --- Hauptfunktion ---
def main():
    num_nodes = 50
    V_max = 5
    R_harmonie = 0.5
    epochs = 10
    learning_rate = 0.1

    # Synthetischer Datensatz für Robotik
    num_samples = 10000
    data = pd.DataFrame({
        'distance': np.random.rand(num_samples),
        'orientation': np.random.rand(num_samples),
        'action': np.random.choice([0, 1], num_samples)  # Beispiel: Binäre Klassifikation (z.B. vorwärts, links)
    })

    # Features und Labels extrahieren
    X = data[['distance', 'orientation']]
    y = data['action']

    # Datensatz aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    nodes = initialize_network(num_nodes, V_max, R_harmonie)
    weights_history = train_network(nodes, X_train.values, epochs, learning_rate)

    # Visualisierung des Netzwerks und der Gewichtshistorie
    visualize_network(nodes)
    visualize_weights_over_time(weights_history)

    # Teste das Netzwerk
    test_network(nodes, X_test.values)

    # Simuliere Roboter-Navigation
    robot_navigation()

if __name__ == "__main__":
    main()
